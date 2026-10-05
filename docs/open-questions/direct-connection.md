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
| HTTPS with byte ranges | Built for the measurement only (`w47`): nginx on a mapped port, a certificate made on the machine for the host's address and read back through ssh, the only one `curl` is told to believe (`--cacert`), a token made for the run. Without the token it answers 403; with the system's certificates alone `curl` refuses it. One stream is twice ssh's (16 to 26 MB/s against 6 to 15); four ranges are 39 to 40 MB/s, which four ssh streams reach once they are up. Not put in the code: at four streams it brings nothing ssh does not, and it is a daemon, a port and a secret more. Worth returning to if more than four streams are wanted, since it does not pass through `MaxStartups` |
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
   four, which leaves the instance's `sshd` room under its ten.
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
