# Reaching a rented machine without Vast's proxy: what it carries and what it may be trusted with

Date: 2026-10-05

Status: measured on two hosts, France and North Carolina; two hosts on the
west coast of the United States were rented for it and never answered, so
that region is not measured, and no host without an open port was tried.
The direct route is in the code with its host keys pinned
(`reverberate.gpu.direct`, `reverberate.gpu.hostkeys`), the proxy stays the
fallback, and the homecoming of `gpu.homecoming` takes the direct route
first. Written for lot L16 of ADR 0016. The measurements are
`python -m reverberate.experiments.w47_direct_connection offers | line | measure`.

Lot L35 (2026-10-07) adds the last section but one, "The pack's way home":
what bounds the rate, the range server that is now in the code, and the
bench that measures every way on a host. **No machine was rented for it**:
what it says is reasoned from the measurements of this note, or measured
on 127.0.0.1 and on the pack of the first scene as it lies on the laptop.

## The answer

A rented machine can be reached at its host's own address, and it is worth
it. From a host in France the direct route brought 41 to 45 MB/s on four
streams where the proxy brought 12, on a line that carries 63; from North
Carolina 21 to 22 against 14, and 40 once the streams were up, on a line
whose one stream to that coast carries 18. Sending from the laptop, the
direct route carried 10 MB/s to France against 3 to 5 through the proxy,
and the same 4 MB/s as the proxy to North Carolina, which is the laptop's
own line that way. A command costs 0.7 s directly against 2.9 s through the
proxy in Europe, and the same 2 s either way across the Atlantic.

The direct address is a new name to `ssh`, and taking the key it shows the
first time would believe a machine in the middle. So the keys are read on
the machine through the proxy, pinned to the address in a file of the
instance's own, and the direct connection refuses any other key. That was
tried on both hosts: a file naming another key is refused with
`Host key verification failed`.

## What Vast offers

An instance is created by `PUT /asks/{offer}/` with a `runtype`. The
values are Vast's command line's own (`vast.py`, `get_runtype`):

| `runtype` | what it gives |
| --- | --- |
| `ssh` | what this repository sent until 2026-10-05: ssh through the proxy `sshN.vast.ai` |
| `ssh_proxy` | the same, as `vastai create instance --ssh` now words it |
| `ssh_direc ssh_proxy` | `--ssh --direct`: the container's port 22 mapped to a port of the host's address, **and** the proxy. The word is `ssh_direc`, without the t |
| `jupyter_direc ssh_direc ssh_proxy` | the same with Jupyter; not used here |

Beside the `runtype`, a key of `env` of the form `"-p 8443:8443": "1"` asks
for one more port of the container on the host (Docker's `-p`). Vast's
documentation allows 64 of them.

Created with `ssh_direc ssh_proxy`, the instance's record
(`GET /instances/{id}/`) read, on both hosts:

| field | value seen | |
| --- | --- | --- |
| `image_runtype` | `ssh_direc ssh_proxy` | |
| `ssh_host`, `ssh_port` | `ssh4.vast.ai`, a port | the proxy, as always, and it answers |
| `public_ipaddr` | the host's address | |
| `ports` | `{"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "50001"}], "8443/tcp": [{..., "HostPort": "50000"}]}` | Docker's map; the host's port is not the container's |
| `direct_port_start`, `direct_port_end` | 50000, 50001 | the range the host gave |
| `extra_env` | `[["-p 8443:8443", "1"], ...]` | |

**Both routes stay open.** Every measurement below used the proxy and the
direct port of the same instance in turn. The record carries the address
as soon as ssh answers through the proxy; on both hosts that was 109 s
after the rental with a small image (`ubuntu:22.04`).

**Which hosts.** An offer says `direct_port_count`, the ports open on the
host's router, and `geolocation` (`"France, FR"`). Of the 248 hosts
offered under 0.25 USD/h on 2026-10-05 with a line over 200 Mbit/s each
way, 220 had 16 ports or more, 26 had 4 to 15, one had 3 and one had none
(Portugal). A host without an open port is the exception and not a class
to plan for. `Offer.direct_ports` reads the count, and
`direct.rank(offers, prefer_regions)` puts hosts with open ports first,
then the regions asked for, keeping the price's order among equals.

What Vast's API does not give is the instance's ssh host key.

## Who is believed, and what is lent

What the two machines showed (`sshd -T`, and `ssh-keyscan` of each route):

- `passwordauthentication no`, `kbdinteractiveauthentication no`,
  `permitrootlogin without-password`: the instance admits keys alone, and
  the key it holds is the account's.
- `allowagentforwarding yes`: the machine would take an agent if one were
  offered. Refusing is the client's to do.
- `maxstartups 10:30:100`, `maxsessions 10`: past ten connections that
  have not authenticated yet, three in ten are dropped. This is what
  refused twelve streams together on the first whole scene.
- **the proxy shows the machine's own host keys**: the three keys
  `ssh-keyscan` reads at `sshN.vast.ai:port` are the three in the
  machine's `/etc/ssh`, and the three at the direct address. The proxy
  relays the connection and does not end it: ssh is end to end on both
  routes.

So, in the code:

1. **Keys only, and the account's only.** Every connection to a pinned or
   a direct machine carries `BatchMode=yes`, `IdentitiesOnly=yes`,
   `PreferredAuthentications=publickey`, `PasswordAuthentication=no`,
   `KbdInteractiveAuthentication=no`. No key is made for a rented host
   and none is copied to it. The one command before the pinning, which
   finds the machine awake, goes as it always did.
2. **Nothing is lent.** `ForwardAgent=no`, `ForwardX11=no`,
   `ClearAllForwardings=yes`, `PermitLocalCommand=no`, `Tunnel=no`, ahead
   of anything `~/.ssh/config` grants. On both machines `SSH_AUTH_SOCK`
   was empty in the session. No reverse tunnel is ever asked for.
3. **The direct address is believed by pinned keys.** When a rented
   machine first answers (`vast.wait_for_ssh`), `direct.upgrade` reads
   `/etc/ssh/ssh_host_*_key.pub` on it through the proxy, writes them to
   `<runs>/known_hosts/instance_<id>` against `[address]:port` (one file
   an instance, readable by the user alone, replaced and never appended
   to), and probes the address with `StrictHostKeyChecking=yes`,
   `UserKnownHostsFile` that file, `GlobalKnownHostsFile=/dev/null`,
   `UpdateHostKeys=no`. The machine handed back still goes through the
   proxy for its commands; its `directly()` is the checked route, and
   that is what `homecoming.fastest` and `rsync` take.
4. **An address that cannot be pinned is dropped**, not used unchecked:
   keys that could not be read, a machine that showed none, an address
   that did not answer, each leaves a machine with no direct address and
   says why. A key that does not match says `DIRECT ROUTE REFUSED`; it is
   the command's failure and is never tried again as a lost connection.
5. **The file goes with the instance** (`vast.teardown`): a host gives
   the same port to its next tenant.

**What is still trusted.** The pinned keys are as good as the connection
they were read through, and the proxy's first connection takes the key it
is shown (`accept-new`, into `~/.ssh/known_hosts`, which holds 236 lines
of them). Someone between the laptop and `sshN.vast.ai` at that moment
could answer in the machine's place; they could not reach the real
machine with the laptop's authentication, which is bound to the session,
so what they could do is show a false machine. Beyond that the trust is
Vast's and the host operator's, who hold the machine whatever the route:
nothing sent to a rented machine is secret from its owner, which is why
nothing of the laptop is lent to it.

**A second channel closes the first connection**, and is in the code but
off (`upgrade(confirm=True)`, `hostkeys.confirmed_by_api`): the machine
writes its keys' fingerprints to its container's output, and the
instance's log is asked of Vast's API over TLS
(`PUT /instances/request_logs/{id}/`). A false machine cannot write to
the real instance's log. On the one host it was tried on, the three
fingerprints came back. It costs one call and some ten seconds; it should
be the default once it has been seen on more hosts.

### The other ways, weighed

| way | verdict |
| --- | --- |
| `rsync` over direct ssh | In the code: `rsync` is given the route's own shell (`direct.rsync_shell`). 21 MB came in 2.3 s directly and 8.6 s through the proxy from France |
| one kept connection (`ControlMaster`) | Kept **one a worker**, as `homecoming` does. A command on a kept connection costs 0.12 s (France) or 0.36 s (North Carolina) against 0.7 to 2.9 s. Four streams on **one** shared connection carried as much as four connections from France (45 MB/s) and little over half from North Carolina (12 against 21): on a long path one TCP connection is one window |
| HTTPS with byte ranges | Built for the measurement only (`w47`): nginx on a mapped port, a certificate made on the machine for the host's address and read back through ssh, the only one `curl` is told to believe (`--cacert`), a token made for the run. Without the token it answers 403; with the system's certificates alone `curl` refuses it. One stream is twice ssh's (16 to 26 MB/s against 6 to 15); four ranges are 39 to 40 MB/s, which four ssh streams reach once they are up. Left out of the code by lot L16: at four streams it brings nothing ssh does not, and it is a daemon, a port and a secret more. **In the code since lot L35**, for more than four streams, since it does not pass through `MaxStartups`: without nginx or any package, and with what it may do written down ("The pack's way home", below) |
| WireGuard, Tailscale | Not built. Either needs a private key or an account's token on the rented machine, which its owner can read and then use to join the laptop's network: the opposite of lending nothing. A container is not usually given a tunnel device, and an overlay adds nothing to a TCP port that is already open |

## What each route carried

One machine a region, each route in turn, both ways. The bytes do not
compress (`/dev/urandom`, written once on the machine). A megabyte is 1e6
bytes. Each measurement moved 50 to 130 MB and lasted 3 to 16 s, so **the
mean includes the connection's handshake and its slow start**; "late" is
the rate over the second half of the measurement's time, and is nearer
what a transfer of gigabytes sees. Where a route was measured twice, both
are given. The laptop was on Wi-Fi and rendering audio meanwhile.

The laptop's own line, one stream to a public file of 100 MB: 63 MB/s to
France, 41 to Germany, 18 to the east of the United States, 12 to the west.

**France** (host 149401, a GTX 1060 at 0.041 USD/h, 6 open ports, proxy `ssh4`):

| route | streams | to the laptop, MB/s (late) | from the laptop, MB/s (late) | first byte, s |
| --- | --- | --- | --- | --- |
| proxy | 1 | 5.0 (6.6), 5.0 (6.6) | 2.9 (3.9) | 3.2 |
| proxy | 4 | 12.2 (23.5), 12.1 (22.7) | 4.8 (5.6) | 3.5 |
| direct ssh | 1 | 15.2 (22.2), 5.7 (6.7) | 9.0 (11.8), 10.2 (12.2) | 0.7 |
| direct ssh | 4 | 13.7 (8.1), 41.2 (56.9) | 11.1 (10.2), 10.5 (8.9) | 0.7 |
| direct ssh, one shared connection | 4 | 44.8 (56.7) | 6.1 (8.3) | 0.2 |
| HTTPS ranges | 1 | 26.1 | 10.5 | 0.2 |
| HTTPS ranges | 4 | 20.7, 40.5 | 10.7 | 0.2 |

**North Carolina** (host 144857, a GTX 1080 at 0.045 USD/h, 256 open ports, proxy `ssh3`):

| route | streams | to the laptop, MB/s (late) | from the laptop, MB/s (late) | first byte, s |
| --- | --- | --- | --- | --- |
| proxy | 1 | 6.1 (9.0) | 3.5 (5.8) | 2.4 |
| proxy | 4 | 14.2 (28.3) | 3.6 (1.9) | 2.3 |
| direct ssh | 1 | 7.9 (9.6), 7.3 (8.9) | 4.0 (5.1), 3.9 (4.7) | 2.2 |
| direct ssh | 4 | 22.0 (43.2), 20.9 (40.1) | 4.8 (1.9), 3.4 (1.4) | 2.2 |
| direct ssh, one shared connection | 4 | 12.5 (16.2) | 2.5 (3.9) | 0.5 |
| HTTPS ranges | 1 | 16.0 | | 0.7 |
| HTTPS ranges | 4 | 39.1 | 4.8 | 0.7 |

A command (`ssh true`), in seconds: France 2.9 through the proxy, 0.7
directly, 0.12 on a kept connection; North Carolina 2.2, 2.0 and 0.36.

How this reads:

- **The measurements are short and they scatter.** From France the direct
  route gave 15.2 then 5.7 MB/s on one stream and 13.7 then 41.2 on four.
  The proxy repeated to a tenth. What is firm is the order, not the
  second figure.
- **The proxies of these two hosts were better than the ones of the two
  whole scenes** (2.0 to 2.3 MB/s a stream, 4.4 on four, from
  California). A relay is drawn with the instance and is not chosen; one
  of them, `ssh7`, fell to 0.05 MB/s after 8 GB, which a measurement of
  100 MB cannot see. The direct route takes that draw out of the run.
- **Sending from the laptop is bounded by the laptop**: 10 to 11 MB/s to
  France on every direct route and by HTTPS alike, 4 to 5 to North
  Carolina on every route.
- **Four streams reach the line.** Late, four direct ssh streams carried
  57 of the 63 MB/s to France and 40 to 43 MB/s to North Carolina, more
  than twice what one stream of the reference file brought there.

## What is recommended

1. **The direct route for every transfer, pinned, the proxy behind it.**
   This is what the code now does without being asked. Commands stay on
   the proxy as they were; in Europe they would gain two seconds each on
   the direct route and nothing across the Atlantic, and a watcher's
   commands are better kept on a connection of their own
   (`Machine.sharing`), which costs a tenth of a second.
2. **Four streams, each on a connection it keeps.** Not one connection
   shared by four, which halves the rate on a long path; not more than
   four, which leaves the instance's `sshd` room under its ten. Since
   lot L35 a transfer by ssh may grow to eight, opened as the count
   doubles and never together, and takes them back at the first refusal;
   what wants more goes in ranges over HTTPS ("The pack's way home").
3. **For a run whose result is large, a host in Europe**:
   `direct.rank(offers, ["FR", "GB", "DE", ...])`, a preference and not a
   filter. A pack of 9.5 GB is 36 minutes at the 4.4 MB/s of the proxy,
   8 minutes at the 20 MB/s measured from North Carolina and 4 at the
   40 measured from France; on a machine at 1.38 USD/h that is 0.83 USD
   against 0.18 or 0.09, and half an hour.
4. **A relay box near the laptop only for a far host and a large
   result.** A machine that bills by the hour can push its result to a
   box in Europe at 0.04 to 0.07 USD/h, at the rate two hosts have
   between them (22 MB/s was seen through proxies on 2026-09-13), and be
   destroyed; the laptop then fetches from Europe. It pays when the far
   host's hour is dear, its own line to the laptop slow, and the result
   over some ten gigabytes. From North Carolina the direct route makes it
   unnecessary. From the west coast it is not known: see below.
5. **The figures a fetch is priced with** (`--line direct` of the trace:
   70 MB/s for Europe, 17 for the United States) can be set by these:
   40 to 57 for France and 20 to 40 for the east coast on four streams.

## The pack's way home (lot L35, 2026-10-07)

No machine was rented for this lot. Each figure below says where it comes
from: the measurements above, 127.0.0.1 (a range server and an `sshd` of the
laptop's own, a line slowed and cut on purpose), or the pack of scene B as
it lies on the laptop. The first rental's bench replaces the reasoning by
figures:

```sh
python -m reverberate.gpu.transfer_bench --instance <the instance the driver just rented>
```

### What limits the rate

| what | is it the limit | by what |
| --- | --- | --- |
| Vast's proxy | it was, and it is out of the path | 2.0 to 2.3 MB/s a stream and 4.4 on four from California; 12 to 14 on four on the two hosts above; 0.05 once, after 8 GB |
| ssh's cipher | no | on 127.0.0.1, both ends on the laptop, one stream carries 477 MB/s with ssh's own choice and four carry 920; four with `aes128-gcm` 681, with `chacha20-poly1305` 340, with `aes128-ctr` 305. Ten times any line measured. Compression is off by default, the bench says so in so many words, and the payload does not deflate (below) |
| one connection shared by the streams (`ControlMaster`) | yes, twice over | one TCP window on a long path, 12.5 MB/s against 21 from North Carolina; and one process's encryption, 275 MB/s on four sessions of one connection on 127.0.0.1 where four connections carry 920. A worker keeps a connection of its own, as since lot L16 |
| one connection's window | yes, on a long path | one ssh stream brought 7 to 8 MB/s from North Carolina where four brought 21 to 22 (40 to 43 late) and one HTTPS stream 16, on a line whose one stream to that coast carries 18. From France one ssh stream is 6 to 15 and four are 41 to 57 |
| how many connections `sshd` takes | yes, for ssh | `MaxStartups 10:30:100`: twelve handshakes at once were refused together. It bounds what is opened at once, which is why four. Ranges over HTTPS do not pass it |
| the machine's line | not known | an offer says `inet_up` and `inet_down`, the host's own measurement of itself. Nothing compared it with what came. **It was not read when an offer was chosen**: the search asked `inet_down > 300` and nothing of `inet_up`, and the prediction caps the fetch at four fifths of `inet_up` under `--line direct` alone |
| the laptop's line | yes, from Europe | 63 to 70 MB/s to France on one stream; four direct ssh streams reached 57 late |

So, from a host in Europe, four direct streams are already at nine tenths
of the laptop's line: nothing makes that part faster, and what is left to
gain is when the transfer starts. From across the Atlantic the bound is a
connection's window, more connections are the remedy, `sshd` refuses them
and a range server does not: four ranges brought 39 to 40 MB/s, and what
sixteen or thirty-two bring is the first bench's to say. The ceiling there
is the laptop's line that way, which no measurement of this note reached.

**The offers.** A floor now stands in the search: an offer must say it
sends 200 Mbit/s (`inet_up`, `gpu.onebox.MIN_INET_UP_MBPS`), 25 MB/s were
it true, under which a pack keeps its machine six minutes and more. Of the
248 hosts counted above every one said more, so the floor costs no offer
today and keeps out the one that would. The preference exists already and
stays as it is: under `--line direct` an offer's fetch is priced at the
lesser of the laptop's line to its region and four fifths of what it says
it sends. It is not the default because no rental has yet shown what
`inet_up` is worth; the bench's table beside the offer's figure is that
comparison.

### What was weighed

| way | verdict |
| --- | --- |
| more direct ssh connections, their count found as the transfer goes | In the code: four to start, eight at most, opened as the count doubles and not together (`transfer.Lanes`). Eight on 127.0.0.1 carry no more than four (792 against 920 MB/s), which is the laptop's processor and says nothing of a line |
| ranges over HTTPS from a server on the machine | In the code, and the first way tried where the host maps the port: sixteen connections to start, thirty-two at most. `gpu/rangeserver.py` is one file for the system's `python3`, sent down the pinned ssh connection; no nginx, no package. On 127.0.0.1 one connection carries 170 MB/s and thirty-two 1100, with the old interpreter of the laptop as the server |
| `rsync` | Kept as the last way, when nothing else brought the file. One stream, so one window: on 127.0.0.1 131 MB/s where `cat` over ssh carries 477, the difference being its own checksums |
| a `tar` stream | Is one ssh stream. It is how the pair cache comes, thousands of files of half a megabyte in batches of 32 MB on four connections (`fetch_tree`), and stays so: ranges serve large files |
| starting before the campaign ends | In the code for what is whole: the machine is asked every 20 s, the pack is brought the moment it is renamed into place, and the campaign's end is seen then. **Not** for a pack that is still being written: see below |
| smaller data | Measured, below: the pack's 16 bit payload does not deflate. Nothing done |
| the store as the middle step | Not weighed: no storage credential goes on a rented machine |

### The range server: what it may do

`reverberate.gpu.rangeserver`, started by `transfer.serve`:

1. **It is started over the pinned ssh connection and no other.** `serve`
   refuses any machine but a `DirectMachine` with its `known_hosts` file.
   The token goes down that connection's stdin, never on a command line;
   the certificate the machine makes comes back on its stdout.
2. **It serves one folder, to read.** `GET` and `HEAD`, whole or one range.
   A path is resolved with its links and served only where the result is a
   regular file under the run's output folder: no listing, and a link that
   points out of the folder is 404. Its own files (the key, the token, the
   log) are in `/root/.rv-serve`, beside the folder and not in it.
3. **To the bearer of the run's token.** 256 bits from `secrets`, made on
   the laptop for each start, compared in constant time before the path is
   looked at. Without it every request is 401 with no body. A server that is
   started again has a new token and a new key; the one before is killed.
4. **Over TLS 1.2 or later, to a client that believes one certificate.** An
   RSA key and a certificate made on the machine by `openssl` for the run.
   The laptop's context holds that certificate as its only authority and
   then compares the SHA-256 of what the server shows with it. A server
   with another certificate is refused as the command's failure, never
   tried again. The certificate names no host: it is believed byte for
   byte, not by a name.
5. **Nothing of it is left.** The key's and the token's files are removed
   once the server has read them; the server ends itself after twelve
   hours, is ended when the fetch is over (`transfer.stop`, which removes
   its folder), and goes with the instance. It holds 96 connections at
   most and closes a handshake that takes more than 15 s.

What that leaves: whoever holds the machine reads the token in the
server's memory, as they read the run's output on its disk. Whoever steals
the token reads the run's output folder until the server ends, and can
write nothing. The digest a fetched file is judged by is asked over ssh,
not of the server: the bytes and their digest come by two channels, both
ending on the same machine.

### How many connections, tried on 127.0.0.1

The server slows its line when told (`--limit-stream`, `--limit-total`) and
cuts bodies (`--drop-after`, `--drops`); a file is then fetched whole by
`fetch_file`, digest and all, so the rates below include the verification.

| the line | workers | came at | |
| --- | --- | --- | --- |
| not slowed, 2 GB | 1, 4, 8 | 375, 473, 471 MB/s | the two digests of the file are the rest |
| 3.6 MB/s a connection, as a window does | 4 | 14.5 MB/s | |
| | 16 | 55.9 MB/s | |
| | 16 to 32, 3 GB | 74.2 MB/s | windows of 60 MB/s on 16 and 115 on 32: doubled, kept |
| the same, and 60 MB/s in all, as a line does | 16 to 32, 3 GB | 51.0 MB/s | windows of 49 MB/s on 16 and 59 on 32: a fifth more, so kept, for a line that 16 nearly fill |
| | 4 to 8, 3 GB | 23.6 MB/s | windows of 13 MB/s on 4 and 26 on 8 |
| 20 MB/s a connection, three bodies cut at 20 MB of 32 | 4 | 31.6 MB/s | three failures, 3 workers after them, the file whole |
| | 16 to 32 | 103.2 MB/s | three failures, 12 workers after them, the file whole |

Through the laptop's own `sshd` (a throwaway one on 127.0.0.1, its key
pinned as a rented machine's is), the whole path ran as it will on a
machine: the server started by `serve` over ssh, its certificate read
back, 2 GB fetched in ranges and verified at 112 MB/s; the same by direct
ssh at 117; a second call for the file brought nothing ("home already").
Three things that would have failed on a rented machine were found there
and are in the code: the server's start waited five seconds on the
resolver for the host's own name, which it never reads; a key on a curve
was refused by an old TLS library, so the key is RSA; a body cut by the
server read as a short chunk and then as a second failure on the dead
connection, so it is now one.

### Before the campaign ends

A rental was looked at every 300 s, so a finished campaign waited 150 s in
the mean for its fetch to start, and 300 at worst: 0.06 to 0.12 USD on the
eight card host, and the wall time. The machine is now asked every 20 s,
on one kept connection (a command there costs 0.12 to 0.36 s), whether
`pack.h5` is there and whether `campaign.done` or `campaign.failed` is.
The pack is fetched at once; the end ends the watcher's pause.

The trace renames its pack into place when it is whole, then checks it. On
the whole scene the check reads the pack back in half a second (`read`),
so the pack comes hardly sooner than the end; on a smoke run the check
renders (`full`) and the pack comes meanwhile. A pack fetched early is not
fetched again: `pack.h5.home.json` beside it says which file of the
machine came, by its size and its SHA-256 there, and a file that changed
on the machine since (a relaunch writes it again) is brought again.

**The pack as it is being written does not come early, and why.** The
write lasts 120 to 153 s, in which 6 to 7.5 GB could come at 50 MB/s. But
a pack that is still open is not yet the pack: HDF5 keeps the index of an
appended dataset in memory and writes it at the close, into places it
reserved all along the file. Tried here on a file written the way the
pack's writer writes (a dataset appended to in chunks of 65 536 values, a
table written a row at a time, six sources, 2.75 GB): of the 80 chunks of
32 MB that a reader saw whole while the file grew, **17 were the same
after the close**. Each of the others differs by a few kilobytes and would
be fetched twice. With the file flushed every 80 MB written, 67 of the 80
are the same. So this needs the writer to flush as it goes
(`render/pack.py`, not this lot's), after which a fetch that follows the
growing file would have five sixths of the pack home when the write ends
and the existing verification would mend the rest. Not done.

### Smaller data

A sample of 500 MB of scene B's `sources/far_2/low/compact/data`, the 16
bit bins that are 6.85 GB of that pack's 8.09 (85 per cent), and its
`tail/moments` (0.89 GB, 11 per cent), under `nice`:

| what | how | of its bytes | packed at | unpacked at |
| --- | --- | --- | --- | --- |
| `low/compact/data`, int16 | zstd -3 | 1.000 | 1262 MB/s | 1652 MB/s |
| | bytes shuffled, zstd -3 | 0.967 | 914 MB/s | 979 MB/s |
| | bytes shuffled, zstd -19 | 0.946 | 24 MB/s | 668 MB/s |
| | bytes shuffled, gzip -1, HDF5's own filter | 0.967 | 20 MB/s | 273 MB/s |
| `tail/moments`, float32, half of it zeros | zstd -3 | 0.353 | 558 MB/s | 658 MB/s |
| | bytes shuffled, zstd -3 | 0.280 | 665 MB/s | 677 MB/s |
| | bytes shuffled, gzip -1, HDF5's own filter | 0.319 | 61 MB/s | 400 MB/s |
| `early/gain`, float32 | bytes shuffled, gzip -1 | 0.224 | 108 MB/s | 608 MB/s |

The bins hold 15.53 bits of information in their 16: the scale of each
row already fills the range. **Nothing is to be had from the pack's
payload**, by any coder at any speed. The tail's moments and the early
gains would give back 0.66 GB of the 8.09 (8 per cent) under HDF5's own
shuffle and gzip, for some fifteen seconds of one core at the write: 13 s
of fetch at 50 MB/s, 33 s at 20. It is the format's to decide
(`docs/formats/scene-pack.md`) and is worth about what it costs; not done.

What else comes home: the reports (7 MB) and, when asked for, the pair
cache, which is larger than the pack (15.2 GB for scene A's 12 371 pairs
as samples; 621 kB a pair in the compact form, 10.5 GB for the scene's
16 887). It comes while the run lasts and never held the machine; a run
that will not trace the dwelling again can leave it (`--no-fetch-pairs`).

### The way up

Scene B's bundle is 3.4 GB: 126 MB of the mirror's prepared scene, 88 MB
of models, and **3.2 GB of pairs carried** from the laptop's cache so that
they are not solved again (2 615 of them). On the laptop's line, 4 to 10
MB/s up, the 215 MB of a first run are half a minute and the carried pairs
five to thirteen, on a machine that is billed and has nothing else to do.
Three things changed:

- **the bundle goes past the proxy** where the machine has an address of
  its own: 10 MB/s to France against 3 to 5 through the proxy, measured
  above. It went through the proxy until now;
- **it goes up beside the provisioning**, which takes minutes of the
  machine's own downloading on another line, and not after it;
- the dwelling's part deflates on the way, as `rsync -z` always did: to
  0.26 of its bytes for the mirror's scene and 0.31 for the
  models (gzip -6).

What the machine could fetch or build itself and does not: the models and
the pairs are on the project's store, which needs a credential the machine
is not given; the models are also public at their source, behind a
licence's token, which is a credential too. The mirror's scene is built
from the models by the laptop in minutes and could be built there; that is
the trace's bundle to change, not the transfer's. The bound that remains
is the laptop's own line up.

### What a pack of 9.5 GB takes

| what the line carries | the pack | at 1.38 USD/h | the limit then |
| --- | --- | --- | --- |
| 4.4 MB/s, the proxy, as the first scene | 36 min | 0.83 USD | the proxy |
| 20 MB/s | 7.9 min | 0.18 USD | the machine's line, or one coast's distance: the bench's table says which |
| 50 MB/s | 3.2 min | 0.07 USD | within a fifth of the laptop's line to Europe (63 to 70 MB/s) |
| 100 MB/s | 1.6 min | 0.04 USD | more than the laptop's line has carried from anywhere: not reached from this laptop, whatever the host sends |

Beside the line, and not shortened by it: the machine's own SHA-256 of the
file, which runs while the chunks come; the laptop's, after the last
chunk, 933 MB/s here, so 10 s for the pack; the server's start,
under two seconds; the first windows of the count of connections, some
ten seconds; and up to 20 s before the end is seen, where it was up to
300. At 50 MB/s the machine is free some four minutes after its last
stage, of which three are the laptop's line.

### What is not verified without a machine

- **That a host maps the port and says so.** Two hosts did for the
  measurement above (`ports["8443/tcp"]` in the instance's record). The
  rental now asks for it every time. Where the record does not name it the
  instance's environment is asked (`VAST_TCP_PORT_8443`), which has never
  been read on a machine. A host that maps none keeps ssh, and says why.
- **That the image holds `python3` and `openssl`.** The driver already runs
  `python3` on a machine before anything is installed; `openssl` comes
  with the image's certificates. Where either is missing the server is not
  started and ssh is the way.
- **Every rate.** Nothing here says what sixteen or thirty-two ranges bring
  from a host, nor that eight ssh connections are admitted where four
  were. The count of workers looks for it as it goes and the bench says it
  in two minutes; the default order (HTTPS, direct ssh, proxy) is a
  reasoned one until then.
- **The looks every 20 s beside a run of hours**: a kept connection that
  lasts, a pack fetched while the trace checks it.
- **The bundle pushed beside the provisioning**: two transfers and the
  machine's own downloads at once, on a host's `sshd`.

## What was not tested

- **The west coast of the United States.** Hosts 148206 (California) and
  18337 (Washington) were rented and neither answered on ssh in ten
  minutes; both were destroyed. Why is not known: the instance's status
  was not read before the teardown, which `w47` now does.
- **A host without an open port.** One was offered (56596, Portugal). On
  such a host the record has no `ports` and the machine stays on the
  proxy; that path is tested with a fake client only. What Vast answers
  to `ssh_direc` there is not known; `VastClient.create` asks again
  without it on HTTP 400.
- **Gigabytes.** Nothing here ran longer than 16 s. The first whole
  pack fetched directly is the measurement that counts.
- **A direct route that dies in the middle of a fetch.** The homecoming
  pauses and tries again on the route it was given; it does not go back
  to the proxy by itself.
- **The second channel on more than one host**, and pinning the proxy's
  own route to the same keys, which the proxy showing the machine's keys
  makes possible and which would end the growth of `~/.ssh/known_hosts`.
- **`w47` in its last form.** It was run as the measurements above, then
  fitted to the `Machine` of the homecoming lot, merged meanwhile; the
  four instances this lot was allowed were spent before it could be run
  again. Its options were read back by the laptop's own `ssh -G`, and
  the pinning is tested against a fake ssh; no connection has yet been
  made through `PinnedMachine.directly()` as it now stands.
