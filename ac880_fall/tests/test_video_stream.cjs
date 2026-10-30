'use strict';
const assert = require('node:assert/strict');
const {MjpegParser, LatestDecoder} = require('../../board_runtime/app/web/video_stream.js');

function part(sequence, payload = Buffer.from([255, 216, sequence, 255, 217])) {
  return Buffer.concat([Buffer.from(`--homeframe\r\nContent-Type: image/jpeg\r\nContent-Length: ${payload.length}\r\nX-Sequence: ${sequence}\r\n\r\n`), payload, Buffer.from('\r\n')]);
}

async function main() {
  const frames = [];
  const parser = new MjpegParser(frame => frames.push(frame));
  const joined = Buffer.concat([part(1), part(2), part(3)]);
  for (const byte of joined) parser.push(Uint8Array.of(byte));
  assert.deepEqual(frames.map(frame => frame.sequence), [1, 2, 3]);
  assert.deepEqual([...frames[2].bytes], [255, 216, 3, 255, 217]);
  const burst = [];
  new MjpegParser(frame => burst.push(frame.sequence)).push(joined);
  assert.deepEqual(burst, [1, 2, 3]);
  assert.throws(() => new MjpegParser(() => {}).push(Buffer.from('--x\r\nContent-Length: 999999999\r\n\r\n')), /长度/);
  assert.throws(() => new MjpegParser(() => {}).push(part(1, Buffer.from('broken'))), /JPEG/);
  assert.throws(() => new MjpegParser(() => {}).push(Buffer.alloc(9000, 65)), /分段头/);

  const jobs = [], releases = [], painted = [], closed = [];
  const decoder = new LatestDecoder(bytes => new Promise(resolve => {
    const id = bytes[2]; jobs.push(id);
    releases.push(() => resolve({id, close: () => closed.push(id)}));
  }), image => painted.push(image.id));
  const offer = id => decoder.offer({bytes: Uint8Array.of(255, 216, id)});
  const tick = () => new Promise(resolve => setImmediate(resolve));
  offer(1); offer(2); offer(3);
  assert.deepEqual(jobs, [1]);
  assert.equal(decoder.skipped, 1);
  releases.shift()(); await tick();
  assert.deepEqual(jobs, [1, 3]);
  releases.shift()(); await tick();
  assert.deepEqual(painted, [1, 3]);
  assert.deepEqual(closed, [1, 3]);
  offer(4); decoder.reset(); offer(5);
  releases.shift()(); await tick();
  assert.deepEqual(painted, [1, 3]);
  releases.shift()(); await tick();
  assert.deepEqual(painted, [1, 3, 5]);
  assert.equal(decoder.busy, false);
  console.log('视频分段、最新帧队列和重连旧帧隔离检查通过。');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
