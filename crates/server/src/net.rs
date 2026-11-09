//! The transport: how messages get from one process to another, or from one
//! half of a process to the other.
//!
//! Block 2.12, designed in `docs/PHASE2_MULTIPLAYER.md` §2 and §5.
//!
//! # One type, two deployments
//!
//! §2's rule, and the reason this module is shaped the way it is:
//!
//! > Anything that is true only because the client and server share a process is
//! > a bug waiting for the socket. The in-process transport must be a real
//! > implementation of the same trait the socket implements — never a shortcut
//! > around it.
//!
//! So there is exactly one type, [`Link`], and it is a pair of channel ends. A
//! local link joins two halves of one process; a TCP link joins two machines.
//! Neither the server nor the client can tell which it has, because there is
//! nothing to tell: both are `Link<Outbound, Inbound>`.
//!
//! That is stronger than a trait with two implementations, which is what the
//! issue for this block asked for. A trait leaves two code paths that must be
//! kept in step; this leaves one, and moves the difference into how the channel
//! ends are wired up. Singleplayer is not a special case of multiplayer here —
//! it is multiplayer with a very short wire.
//!
//! # Why `std::sync::mpsc` and `std::net`
//!
//! No dependency, per the owner's standing constraint. `mpsc` gives the
//! non-blocking mailbox a tick loop wants, and TCP gives reliable ordered
//! delivery, which suits a protocol where a missed edit is a desync rather than
//! a dropped frame. UDP with a reliability layer is the better endpoint under
//! packet loss and is its own project (`RESEARCH_MULTIPLAYER.md` §3.2); it can
//! be added as another way of wiring a `Link` without anything above here
//! noticing.
//!
//! # What is deliberately not here
//!
//! **No authentication and no encryption.** A `Link` will carry whatever it is
//! given, to whoever connected. Untrusted clients are block 2.14, and saying so
//! plainly is better than a comment implying this is ready for the open
//! internet. It is not.

use std::io::{Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender, TryRecvError};
use std::sync::Arc;

use crate::wire::{frame, unframe, ClientMessage, ServerMessage, WireError};

/// How many messages may sit unwritten in one client's outbound queue before
/// that client is dropped.
///
/// A client that stops reading fills its socket buffers, the writer thread
/// blocks in `write_all`, and without a bound the tick loop keeps queueing
/// into memory forever. A healthy client's queue is a handful of messages
/// (two per tick, drained as fast as the socket takes them), so this is about
/// a minute of total silence at 60 tps -- far beyond a lag spike, far below a
/// memory problem.
pub const MAX_QUEUED_MESSAGES: usize = 4096;

/// How often a blocked reader wakes to check whether its link was shut down.
const READ_POLL: std::time::Duration = std::time::Duration::from_millis(200);

/// The limits an [`Acceptor`] and the links it hands out enforce.
///
/// Tests construct smaller ones so that hitting a limit does not take
/// thousands of ticks. The session's own numbers are in `headless.rs`.
#[derive(Clone, Copy, Debug)]
pub struct NetLimits {
    /// Most connections open at once (pending + seated). Excess connections
    /// are closed at accept, before any thread is spawned for them.
    pub max_connections: usize,
    /// Most unwritten messages one client may have queued
    /// ([`MAX_QUEUED_MESSAGES`]).
    pub max_queued: usize,
}

impl Default for NetLimits {
    fn default() -> Self {
        Self {
            max_connections: usize::MAX,
            max_queued: MAX_QUEUED_MESSAGES,
        }
    }
}

/// Decrements the acceptor's open-connection count when the link is dropped.
struct ConnSlot(Arc<AtomicUsize>);

impl Drop for ConnSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Sets a flag when dropped: how the reader thread says it has ended.
struct Done(Arc<AtomicBool>);

impl Drop for Done {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// What a socket-backed link holds beyond its channel ends.
struct TcpState {
    /// A handle on the socket, kept so the link can shut it down: the reader
    /// thread blocks in `read`, and only closing the socket wakes it.
    stream: TcpStream,
    /// Messages sent but not yet written to the socket.
    queued: Arc<AtomicUsize>,
    budget: usize,
    /// Set by the reader thread as it exits.
    reader_done: Arc<AtomicBool>,
    /// Asks the reader thread to end. A socket shutdown *usually* wakes a
    /// blocked `read`, but not on every platform for a duplicated handle, so
    /// the reader also polls this on a short read timeout.
    stop: Arc<AtomicBool>,
    slot: Option<ConnSlot>,
}

/// One end of a conversation: send `S`, receive `R`.
///
/// The server holds `Link<ServerMessage, ClientMessage>`; a client holds the
/// mirror image. Nothing about the type says whether the other end is a function
/// call away or a network away, which is the point.
pub struct Link<S, R> {
    out: Sender<S>,
    inbox: Receiver<R>,
    /// Set once the far end has gone. Sticky: a link does not come back.
    closed: bool,
    /// `None` for an in-process link.
    tcp: Option<TcpState>,
}

impl<S, R> Drop for Link<S, R> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl<S, R> Link<S, R> {
    /// Queue a message. Never blocks, and never fails loudly: a send to a link
    /// whose far end has gone marks it closed and drops the message, which is
    /// what actually happened.
    ///
    /// On a socket, a client whose backlog has reached its budget is closed
    /// (and logged) instead of queued for: it is not reading.
    pub fn send(&mut self, msg: S) {
        if self.closed {
            return;
        }
        if let Some(t) = &self.tcp {
            if t.queued.load(Ordering::SeqCst) >= t.budget {
                log::warn!(
                    "dropping a client that is not reading: {} messages unwritten",
                    t.budget
                );
                self.shutdown();
                return;
            }
            t.queued.fetch_add(1, Ordering::SeqCst);
        }
        if self.out.send(msg).is_err() {
            self.closed = true;
        }
    }

    /// Close the link now: mark it closed and, on a socket, shut it down in
    /// both directions so the blocking reader thread returns and the file
    /// descriptor is released. Idempotent; also what dropping the link does.
    pub fn shutdown(&mut self) {
        self.closed = true;
        if let Some(t) = &self.tcp {
            t.stop.store(true, Ordering::SeqCst);
            let _ = t.stream.shutdown(Shutdown::Both);
        }
    }

    /// Whether the socket reader thread is still running. Always `false` for
    /// an in-process link, which has none.
    pub fn reader_running(&self) -> bool {
        self.tcp
            .as_ref()
            .is_some_and(|t| !t.reader_done.load(Ordering::SeqCst))
    }

    /// Everything that has arrived since the last call. Never blocks.
    ///
    /// Returning a `Vec` rather than one message at a time is deliberate: a tick
    /// consumes a tick's worth of input, and a loop that took one message per
    /// tick would fall further behind the faster a client sent.
    pub fn poll(&mut self) -> Vec<R> {
        let mut got = Vec::new();
        loop {
            match self.inbox.try_recv() {
                Ok(m) => got.push(m),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.closed = true;
                    break;
                }
            }
        }
        got
    }

    /// Whether the far end has gone. Only ever observed through [`send`] or
    /// [`poll`]: a channel does not report a disconnect until it is asked.
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

/// Two links wired to each other, in one process.
///
/// What singleplayer runs on. The round trip is a channel push, so it is
/// effectively free — and it goes through the same `send`/`poll` a socket does,
/// so nothing above can accidentally depend on it being free.
pub fn local_pair() -> (
    Link<ServerMessage, ClientMessage>,
    Link<ClientMessage, ServerMessage>,
) {
    let (to_client, from_server) = channel();
    let (to_server, from_client) = channel();
    (
        Link {
            out: to_client,
            inbox: from_client,
            closed: false,
            tcp: None,
        },
        Link {
            out: to_server,
            inbox: from_server,
            closed: false,
            tcp: None,
        },
    )
}

/// Wire a `Link` onto a TCP stream.
///
/// Two threads per connection: one reading frames off the socket into the
/// inbox, one draining the outbox onto it. The threads exist so neither `send`
/// nor `poll` can block the tick loop — a client on a slow connection must not
/// be able to stall the world for everyone else, which it would if the server
/// wrote to its socket inline.
///
/// Both threads end when their channel or the socket does, so dropping the
/// `Link` cleans up without a shutdown protocol.
fn wire_stream<S, R>(
    stream: TcpStream,
    encode: fn(&S) -> Vec<u8>,
    decode: fn(&[u8]) -> Result<R, WireError>,
    budget: usize,
) -> std::io::Result<Link<S, R>>
where
    S: Send + 'static,
    R: Send + 'static,
{
    // Nagle off: these are small messages on a tick clock, and coalescing them
    // trades latency for a bandwidth saving that does not matter here.
    stream.set_nodelay(true)?;
    let reader = stream.try_clone()?;
    let handle = stream.try_clone()?;
    let mut writer = stream;
    let queued = Arc::new(AtomicUsize::new(0));
    let writer_queued = Arc::clone(&queued);
    let reader_done = Arc::new(AtomicBool::new(false));
    let reader_flag = Arc::clone(&reader_done);
    let stop = Arc::new(AtomicBool::new(false));
    let reader_stop = Arc::clone(&stop);
    reader.set_read_timeout(Some(READ_POLL))?;

    let (inbound_tx, inbox) = channel::<R>();
    let (out, outbound_rx) = channel::<S>();

    std::thread::spawn(move || {
        // However this thread ends, say so.
        let _done = Done(reader_flag);
        let mut reader = reader;
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break, // the far end hung up
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
                Err(e)
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                    ) =>
                {
                    if reader_stop.load(Ordering::SeqCst) {
                        break;
                    }
                    continue;
                }
                Err(e) => {
                    log::debug!("connection read failed: {e}");
                    break;
                }
            }
            // A read is not a message: it may hold several, or half of one.
            loop {
                match unframe(&buf) {
                    Ok(Some((payload, used))) => {
                        match decode(payload) {
                            Ok(msg) => {
                                if inbound_tx.send(msg).is_err() {
                                    return; // nobody is listening any more
                                }
                            }
                            Err(e) => {
                                // Malformed from the far end. Drop the
                                // connection rather than guess: block 2.14 will
                                // have more to say about who gets to send this.
                                log::warn!("undecodable message, closing connection: {e}");
                                return;
                            }
                        }
                        buf.drain(..used);
                    }
                    Ok(None) => break, // the rest is still in flight
                    Err(e) => {
                        log::warn!("bad framing, closing connection: {e}");
                        return;
                    }
                }
            }
        }
    });

    std::thread::spawn(move || {
        let mut framed = Vec::new();
        while let Ok(msg) = outbound_rx.recv() {
            framed.clear();
            frame(&encode(&msg), &mut framed);
            writer_queued.fetch_sub(1, Ordering::SeqCst);
            if writer.write_all(&framed).is_err() {
                // Wake the reader too: a link that cannot write is finished.
                let _ = writer.shutdown(Shutdown::Both);
                break;
            }
        }
        let _ = writer.flush();
    });

    Ok(Link {
        out,
        inbox,
        closed: false,
        tcp: Some(TcpState {
            stream: handle,
            queued,
            budget,
            reader_done,
            stop,
            slot: None,
        }),
    })
}

fn encode_server(m: &ServerMessage) -> Vec<u8> {
    let mut b = Vec::new();
    m.encode(&mut b);
    b
}

fn encode_client(m: &ClientMessage) -> Vec<u8> {
    let mut b = Vec::new();
    m.encode(&mut b);
    b
}

/// The server's side of one accepted connection.
pub fn serve(stream: TcpStream) -> std::io::Result<Link<ServerMessage, ClientMessage>> {
    serve_with(stream, MAX_QUEUED_MESSAGES)
}

fn serve_with(
    stream: TcpStream,
    budget: usize,
) -> std::io::Result<Link<ServerMessage, ClientMessage>> {
    wire_stream(stream, encode_server, ClientMessage::decode, budget)
}

/// A client's side, connected to `addr`.
pub fn connect(addr: impl ToSocketAddrs) -> std::io::Result<Link<ClientMessage, ServerMessage>> {
    wire_stream(
        TcpStream::connect(addr)?,
        encode_client,
        ServerMessage::decode,
        // A client's own queue is bounded by what a person can type; only the
        // server has a stranger on the other end.
        usize::MAX,
    )
}

/// A listener that hands over connections without blocking the tick loop.
///
/// `accept` on a `TcpListener` blocks, and a world must keep turning whether or
/// not anyone is knocking. So the accepting happens on its own thread and
/// arrives here as a queue the loop drains once a tick — the same shape as
/// everything else in this module, for the same reason.
pub struct Acceptor {
    incoming: Receiver<Link<ServerMessage, ClientMessage>>,
    addr: std::net::SocketAddr,
    refused: Arc<AtomicU64>,
}

impl Acceptor {
    /// Listen with default limits (no connection cap).
    pub fn bind(addr: impl ToSocketAddrs) -> std::io::Result<Self> {
        Self::bind_with(addr, NetLimits::default())
    }

    /// Listen, refusing connections beyond `limits.max_connections` at accept
    /// -- closed before a thread is spawned for them.
    pub fn bind_with(addr: impl ToSocketAddrs, limits: NetLimits) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let addr = listener.local_addr()?;
        let (tx, incoming) = channel();
        let open = Arc::new(AtomicUsize::new(0));
        let refused = Arc::new(AtomicU64::new(0));
        let refused_in = Arc::clone(&refused);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                if open.load(Ordering::SeqCst) >= limits.max_connections {
                    refused_in.fetch_add(1, Ordering::SeqCst);
                    log::warn!(
                        "refusing a connection: {} already open",
                        limits.max_connections
                    );
                    let _ = stream.shutdown(Shutdown::Both);
                    continue;
                }
                match serve_with(stream, limits.max_queued) {
                    Ok(mut link) => {
                        open.fetch_add(1, Ordering::SeqCst);
                        if let Some(t) = link.tcp.as_mut() {
                            t.slot = Some(ConnSlot(Arc::clone(&open)));
                        }
                        if tx.send(link).is_err() {
                            return; // the server has shut down
                        }
                    }
                    Err(e) => log::warn!("could not set up a connection: {e}"),
                }
            }
        });
        Ok(Self {
            incoming,
            addr,
            refused,
        })
    }

    /// Connections closed at accept for exceeding the connection cap.
    pub fn refused(&self) -> u64 {
        self.refused.load(Ordering::SeqCst)
    }

    /// The address actually bound. Not the one asked for: a test binds port 0
    /// and needs to be told which port it got.
    pub fn addr(&self) -> std::net::SocketAddr {
        self.addr
    }

    /// Whoever has connected since the last call.
    pub fn accepted(&mut self) -> Vec<Link<ServerMessage, ClientMessage>> {
        let mut got = Vec::new();
        while let Ok(link) = self.incoming.try_recv() {
            got.push(link);
        }
        got
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubara_sim::PlayerId;

    /// The local link is a real link: messages go both ways through the same
    /// `send`/`poll` a socket uses.
    #[test]
    fn a_local_pair_carries_messages_both_ways() {
        let (mut server, mut client) = local_pair();

        client.send(ClientMessage::Hello([1, 2, 3]));
        assert_eq!(server.poll(), vec![ClientMessage::Hello([1, 2, 3])]);

        server.send(ServerMessage::Welcome {
            seed: 7,
            you: PlayerId(0),
            blocks: 1,
            items: 2,
        });
        assert_eq!(
            client.poll(),
            vec![ServerMessage::Welcome {
                seed: 7,
                you: PlayerId(0),
                blocks: 1,
                items: 2,
            }]
        );
    }

    /// Polling an empty link is not an error and does not block. A tick calls
    /// this every time round whether or not anything arrived.
    #[test]
    fn polling_an_idle_link_returns_nothing_and_does_not_block() {
        let (mut server, _client) = local_pair();
        assert!(server.poll().is_empty());
        assert!(!server.is_closed());
    }

    #[test]
    fn a_link_notices_when_the_far_end_goes_away() {
        let (mut server, client) = local_pair();
        drop(client);
        let _ = server.poll();
        assert!(server.is_closed(), "the server did not notice the hang-up");
    }

    /// Over a real socket, on localhost. Two threads rather than two processes
    /// -- the two-*process* test is `tests/two_processes.rs`, which is the gate
    /// criterion; this one is here to catch a framing bug without paying for a
    /// process spawn.
    #[test]
    fn a_tcp_link_carries_the_same_messages_a_local_one_does() {
        let mut acceptor = Acceptor::bind("127.0.0.1:0").expect("bind");
        let addr = acceptor.addr();

        let mut client = connect(addr).expect("connect");
        client.send(ClientMessage::Hello([1, 2, 3]));

        // Accepting and delivery are on other threads, so this is the one place
        // the test has to wait for something.
        let mut server = None;
        for _ in 0..200 {
            if let Some(link) = acceptor.accepted().into_iter().next() {
                server = Some(link);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let mut server = server.expect("the connection was accepted");

        let mut got = Vec::new();
        for _ in 0..200 {
            got = server.poll();
            if !got.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(
            got,
            vec![ClientMessage::Hello([1, 2, 3])],
            "over a real socket"
        );
    }

    /// A batch bigger than one `read` must come back whole, in order. This is
    /// the bug the framing exists to prevent, and the one that only appears
    /// under load if it is not tested for.
    #[test]
    fn many_messages_at_once_survive_the_stream() {
        let mut acceptor = Acceptor::bind("127.0.0.1:0").expect("bind");
        let mut client = connect(acceptor.addr()).expect("connect");

        const N: u64 = 500;
        for i in 0..N {
            client.send(ClientMessage::Act(crate::Action::ClickFurnace {
                pos: [i as i32, 0, 0],
                slot: crate::FurnaceSlot::Input,
            }));
        }

        let mut server = None;
        for _ in 0..200 {
            if let Some(link) = acceptor.accepted().into_iter().next() {
                server = Some(link);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let mut server = server.expect("accepted");

        let mut all = Vec::new();
        for _ in 0..400 {
            all.extend(server.poll());
            if all.len() as u64 >= N {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        assert_eq!(all.len() as u64, N, "messages were lost across reads");
        for (i, m) in all.iter().enumerate() {
            let crate::wire::ClientMessage::Act(crate::Action::ClickFurnace { pos, .. }) = m else {
                panic!("wrong message came back: {m:?}");
            };
            assert_eq!(pos[0], i as i32, "messages arrived out of order");
        }
    }

    fn accept_one(acceptor: &mut Acceptor) -> Link<ServerMessage, ClientMessage> {
        for _ in 0..400 {
            if let Some(link) = acceptor.accepted().into_iter().next() {
                return link;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("nothing was accepted");
    }

    /// `shutdown` ends the blocking reader thread and closes the socket; the
    /// peer, which sent nothing, is the witness for the second half.
    #[test]
    fn shutting_a_link_down_ends_its_reader_thread() {
        use std::io::Read;
        let mut acceptor = Acceptor::bind("127.0.0.1:0").expect("bind");
        let mut peer = std::net::TcpStream::connect(acceptor.addr()).expect("connect");
        let mut server = accept_one(&mut acceptor);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            server.reader_running(),
            "the reader was not running to begin with"
        );

        server.shutdown();

        let mut ended = false;
        for _ in 0..400 {
            if !server.reader_running() {
                ended = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(ended, "the reader thread outlived the shutdown");
        assert!(server.is_closed());
        peer.set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        assert_eq!(peer.read(&mut [0u8; 8]).expect("a clean EOF"), 0);
    }

    /// Dropping a link closes its socket too: nobody has to remember to call
    /// `shutdown`.
    #[test]
    fn dropping_a_link_closes_its_socket() {
        use std::io::Read;
        let mut acceptor = Acceptor::bind("127.0.0.1:0").expect("bind");
        let mut peer = std::net::TcpStream::connect(acceptor.addr()).expect("connect");
        let server = accept_one(&mut acceptor);
        drop(server);
        peer.set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        assert_eq!(peer.read(&mut [0u8; 8]).expect("a clean EOF"), 0);
    }

    /// Connections past the cap are closed at accept, and a slot is freed when
    /// a link is dropped.
    #[test]
    fn the_acceptor_refuses_connections_past_its_cap() {
        use std::io::Read;
        let mut acceptor = Acceptor::bind_with(
            "127.0.0.1:0",
            NetLimits {
                max_connections: 2,
                ..NetLimits::default()
            },
        )
        .expect("bind");
        let mut peers: Vec<_> = (0..4)
            .map(|_| std::net::TcpStream::connect(acceptor.addr()).expect("connect"))
            .collect();
        let mut links = Vec::new();
        for _ in 0..200 {
            links.extend(acceptor.accepted());
            if acceptor.refused() >= 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(links.len(), 2, "the cap let a different number through");
        assert_eq!(acceptor.refused(), 2);
        let mut closed = 0;
        for p in &mut peers {
            p.set_read_timeout(Some(std::time::Duration::from_millis(300)))
                .unwrap();
            if matches!(p.read(&mut [0u8; 8]), Ok(0)) {
                closed += 1;
            }
        }
        assert_eq!(closed, 2, "refused connections were left open");

        // A freed slot is reusable.
        links.pop();
        let _again = std::net::TcpStream::connect(acceptor.addr()).expect("connect");
        let mut got = Vec::new();
        for _ in 0..200 {
            got.extend(acceptor.accepted());
            if !got.is_empty() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(got.len(), 1, "a dropped link did not free its slot");
    }

    /// A peer that never reads: past the budget the link closes itself instead
    /// of queueing without bound.
    #[test]
    fn a_link_to_a_peer_that_never_reads_closes_at_its_budget() {
        let mut acceptor = Acceptor::bind_with(
            "127.0.0.1:0",
            NetLimits {
                max_queued: 16,
                ..NetLimits::default()
            },
        )
        .expect("bind");
        let _peer = std::net::TcpStream::connect(acceptor.addr()).expect("connect");
        let mut server = accept_one(&mut acceptor);

        let mut sent = 0u64;
        while !server.is_closed() && sent < 50_000_000 {
            server.send(ServerMessage::Tick(sent));
            sent += 1;
        }
        assert!(
            server.is_closed(),
            "{sent} messages queued and it never closed"
        );
    }
}
