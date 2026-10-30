/* 视频解码只保留一个在途帧和一个最新待处理帧。 */
(function (root) {
  'use strict';

  class MjpegParser {
    constructor(onFrame, maxBytes = 5000000) {
      this.onFrame = onFrame;
      this.maxBytes = maxBytes;
      this.buffer = new Uint8Array(0);
      this.fields = null;
      this.length = 0;
    }

    push(chunk) {
      if (this.buffer.length + chunk.length > this.maxBytes + 8192) {
        throw Error('视频分段超过大小限制');
      }
      const joined = new Uint8Array(this.buffer.length + chunk.length);
      joined.set(this.buffer);
      joined.set(chunk, this.buffer.length);
      this.buffer = joined;
      while (true) {
        if (!this.fields) {
          let end = -1;
          for (let i = 0; i + 3 < this.buffer.length; i++) {
            if (this.buffer[i] === 13 && this.buffer[i + 1] === 10 &&
                this.buffer[i + 2] === 13 && this.buffer[i + 3] === 10) {
              end = i;
              break;
            }
          }
          if (end < 0) {
            if (this.buffer.length > 8192) throw Error('视频分段头无效');
            return;
          }
          const header = new TextDecoder().decode(this.buffer.subarray(0, end));
          const fields = {};
          for (const line of header.split('\r\n')) {
            const colon = line.indexOf(':');
            if (colon > 0) fields[line.slice(0, colon).toLowerCase()] = line.slice(colon + 1).trim();
          }
          const length = Number(fields['content-length']);
          if (!Number.isSafeInteger(length) || length < 4 || length > this.maxBytes) {
            throw Error('视频分段长度无效');
          }
          this.fields = fields;
          this.length = length;
          this.buffer = this.buffer.subarray(end + 4);
        }
        if (this.buffer.length < this.length) return;
        const bytes = this.buffer.slice(0, this.length);
        if (bytes[0] !== 255 || bytes[1] !== 216 ||
            bytes[bytes.length - 2] !== 255 || bytes[bytes.length - 1] !== 217) {
          throw Error('视频帧不是完整 JPEG');
        }
        const fields = this.fields;
        this.buffer = this.buffer.subarray(this.length);
        this.fields = null;
        this.onFrame({bytes, sequence: Number(fields['x-sequence'])});
      }
    }
  }

  class LatestDecoder {
    constructor(decode, paint, onError = () => {}) {
      this.decode = decode;
      this.paint = paint;
      this.onError = onError;
      this.latest = null;
      this.busy = false;
      this.generation = 0;
      this.skipped = 0;
    }

    offer(frame) {
      if (this.latest) this.skipped++;
      this.latest = {frame, generation: this.generation};
      if (!this.busy) this.pump();
    }

    reset() {
      this.generation++;
      this.latest = null;
    }

    async pump() {
      this.busy = true;
      try {
        while (this.latest) {
          const item = this.latest;
          this.latest = null;
          let image;
          try {
            image = await this.decode(item.frame.bytes);
            if (item.generation === this.generation) this.paint(image, item.frame);
          } catch (error) {
            if (item.generation === this.generation) this.onError(error);
          } finally {
            if (image && image.close) image.close();
          }
        }
      } finally {
        this.busy = false;
      }
    }
  }

  async function decodeJpeg(bytes) {
    const blob = new Blob([bytes], {type: 'image/jpeg'});
    if (typeof root.createImageBitmap === 'function') return root.createImageBitmap(blob);
    return new Promise((resolve, reject) => {
      const image = new Image();
      const url = URL.createObjectURL(blob);
      const timer = setTimeout(() => { URL.revokeObjectURL(url); reject(Error('视频解码超时')); }, 3000);
      image.onload = () => { clearTimeout(timer); URL.revokeObjectURL(url); resolve(image); };
      image.onerror = () => { clearTimeout(timer); URL.revokeObjectURL(url); reject(Error('视频解码失败')); };
      image.src = url;
    });
  }

  class VideoPlayer {
    constructor(canvas, fallback, onFrame, onStatus) {
      this.canvas = canvas;
      this.fallback = fallback;
      this.onFrame = onFrame;
      this.onStatus = onStatus;
      this.controller = null;
      this.retry = null;
      this.active = false;
      this.generation = 0;
      this.times = [];
      this.lastPaint = 0;
      this.decoder = new LatestDecoder(decodeJpeg, (image) => {
        if (!this.active) return;
        if (canvas.width !== image.width || canvas.height !== image.height) {
          canvas.width = image.width;
          canvas.height = image.height;
        }
        canvas.getContext('2d', {alpha: false}).drawImage(image, 0, 0);
        this.lastPaint = performance.now();
        this.times.push(this.lastPaint);
        if (this.times.length > 60) this.times.shift();
        onFrame(this.stats());
      }, () => this.onStatus('视频解码失败，正在重连'));
    }

    stats() {
      const fresh = performance.now() - this.lastPaint < 2000;
      const times = this.times;
      return {
        fresh: Boolean(this.lastPaint && fresh),
        fps: fresh && times.length > 1 ? (times.length - 1) * 1000 / (times[times.length - 1] - times[0]) : 0,
        width: this.canvas.width,
        height: this.canvas.height,
        skipped: this.decoder.skipped,
        native: this.native,
      };
    }

    start() {
      if (this.active) return;
      this.active = true;
      this.native = !root.ReadableStream || !root.AbortController || !root.fetch;
      if (this.native) {
        this.canvas.hidden = true;
        this.fallback.hidden = false;
        this.fallback.src = '/stream.mjpg?t=' + Date.now();
        this.fallback.onerror = () => {
          if (this.active) this.retry = setTimeout(() => {
            this.fallback.src = '/stream.mjpg?t=' + Date.now();
          }, 2000);
        };
        return;
      }
      this.canvas.hidden = false;
      this.fallback.hidden = true;
      this.connect(++this.generation);
    }

    stop() {
      this.active = false;
      this.generation++;
      clearTimeout(this.retry);
      if (this.controller) this.controller.abort();
      this.decoder.reset();
      this.times = [];
      this.lastPaint = 0;
      this.fallback.removeAttribute('src');
    }

    async connect(generation) {
      let reader;
      const controller = new AbortController();
      this.controller = controller;
      try {
        const response = await fetch('/stream.mjpg', {
          cache: 'no-store', credentials: 'same-origin', signal: controller.signal,
        });
        if (!response.ok || !response.body) throw Error('视频连接不可用');
        reader = response.body.getReader();
        const parser = new MjpegParser(frame => this.decoder.offer(frame));
        while (this.active && generation === this.generation) {
          const item = await reader.read();
          if (!this.active || generation !== this.generation) break;
          if (item.done) throw Error('视频连接已结束');
          parser.push(item.value);
        }
      } catch (error) {
        if (this.active && generation === this.generation && error.name !== 'AbortError') {
          this.onStatus('视频连接中断，正在重连');
        }
      } finally {
        if (reader) reader.cancel().catch(() => {});
        controller.abort();
        if (this.active && generation === this.generation) {
          this.decoder.reset();
          this.retry = setTimeout(() => this.connect(generation), 1000);
        }
      }
    }
  }

  const api = {MjpegParser, LatestDecoder, VideoPlayer};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.HomeVideo = api;
})(typeof window !== 'undefined' ? window : globalThis);
