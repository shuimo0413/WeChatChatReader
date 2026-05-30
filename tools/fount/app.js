(function () {
  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const inpImages = document.getElementById("inpImages");
  const inpLabels = document.getElementById("inpLabels");
  const boxList = document.getElementById("boxList");
  const imgIdxEl = document.getElementById("imgIdx");
  const canvasHint = document.getElementById("canvasHint");
  const classSeg = document.getElementById("classSeg");
  const statusLine = document.getElementById("statusLine");

  let images = [];
  let cur = -1;
  let labelsByName = {};
  let currentClass = 0;
  let selectedBoxIndex = -1;

  let imgEl = null;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let naturalW = 0;
  let naturalH = 0;

  let drawing = false;
  let startNat = null;

  function colorForCls(cls) {
    if (cls === 0) return "#3fb950";
    if (cls === 1) return "#d29922";
    if (cls === 2) return "#c084fc";
    return "#8b949e";
  }

  function setStatus(text, cls) {
    statusLine.textContent = text || "";
    statusLine.className = "status" + (cls ? " " + cls : "");
  }

  function dispToNat(x, y) {
    return {
      x: (x - offsetX) / scale,
      y: (y - offsetY) / scale,
    };
  }

  function clampNat(x, y) {
    return {
      x: Math.max(0, Math.min(naturalW, x)),
      y: Math.max(0, Math.min(naturalH, y)),
    };
  }

  function getBoxes() {
    const name = images[cur]?.name;
    if (!name) return [];
    if (!labelsByName[name]) labelsByName[name] = [];
    return labelsByName[name];
  }

  function layoutCanvas() {
    const wrap = cv.parentElement;
    const maxW = Math.max(320, wrap.clientWidth - 32);
    const maxH = Math.max(240, window.innerHeight - 120);
    if (!imgEl || !imgEl.complete || !naturalW) {
      cv.width = maxW;
      cv.height = 400;
      return;
    }
    const s = Math.min(maxW / naturalW, maxH / naturalH, 1);
    scale = s;
    const dw = naturalW * s;
    const dh = naturalH * s;
    cv.width = Math.floor(dw);
    cv.height = Math.floor(dh);
    offsetX = 0;
    offsetY = 0;
  }

  function draw() {
    if (!imgEl || !naturalW) return;
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(imgEl, 0, 0, cv.width, cv.height);

    const boxes = getBoxes();
    boxes.forEach((b, i) => {
      const x1 = b.x1 * scale;
      const y1 = b.y1 * scale;
      const x2 = b.x2 * scale;
      const y2 = b.y2 * scale;
      const w = x2 - x1;
      const h = y2 - y1;
      const col = colorForCls(b.cls);
      ctx.strokeStyle = i === selectedBoxIndex ? "#3d8bfd" : col;
      ctx.lineWidth = i === selectedBoxIndex ? 3 : 2;
      ctx.strokeRect(x1, y1, w, h);
      ctx.fillStyle = i === selectedBoxIndex ? "rgba(61,139,253,0.15)" : "rgba(255,255,255,0.04)";
      ctx.fillRect(x1, y1, w, h);
      ctx.fillStyle = col;
      ctx.font = "12px sans-serif";
      ctx.fillText(String(b.cls), x1 + 4, y1 + 14);
    });
  }

  function refreshBoxList() {
    boxList.innerHTML = "";
    const boxes = getBoxes();
    boxes.forEach((b, i) => {
      const li = document.createElement("li");
      li.className = "cls-" + b.cls + (i === selectedBoxIndex ? " selected" : "");
      li.textContent = `类${b.cls} #${i + 1}  (${Math.round(b.x1)},${Math.round(b.y1)})`;
      const del = document.createElement("button");
      del.type = "button";
      del.className = "btn small secondary";
      del.textContent = "删";
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        boxes.splice(i, 1);
        selectedBoxIndex = -1;
        refreshBoxList();
        draw();
      });
      li.appendChild(del);
      li.addEventListener("click", () => {
        selectedBoxIndex = i;
        refreshBoxList();
        draw();
      });
      boxList.appendChild(li);
    });
  }

  function updateIdx() {
    imgIdxEl.textContent = images.length ? `${cur + 1} / ${images.length}` : "0 / 0";
  }

  function loadImageAt(index) {
    if (index < 0 || index >= images.length) return;
    cur = index;
    selectedBoxIndex = -1;
    const item = images[cur];
    canvasHint.classList.add("hidden");

    imgEl = new Image();
    imgEl.onload = () => {
      naturalW = imgEl.naturalWidth;
      naturalH = imgEl.naturalHeight;
      const curIm = images[cur];
      if (curIm) {
        curIm.w = naturalW;
        curIm.h = naturalH;
      }
      layoutCanvas();
      draw();
      refreshBoxList();
      updateIdx();
    };
    imgEl.onerror = () => {
      canvasHint.classList.remove("hidden");
      canvasHint.textContent = "图片加载失败";
    };
    imgEl.src = item.url;
  }

  function revokeBlobUrls() {
    images.forEach((x) => {
      if (x.url && x.url.startsWith("blob:")) URL.revokeObjectURL(x.url);
    });
  }

  async function loadListFromServer() {
    setStatus("", "");
    try {
      const r = await fetch("/api/list-before");
      if (!r.ok) throw new Error("HTTP " + r.status);
      const names = await r.json();
      revokeBlobUrls();
      images = names.map((name) => ({
        name,
        url: "/api/image/before?name=" + encodeURIComponent(name),
        server: true,
      }));
      labelsByName = {};
      cur = -1;
      if (images.length) {
        loadImageAt(0);
        setStatus("已从 before_img 加载 " + images.length + " 张", "ok");
      } else {
        canvasHint.classList.remove("hidden");
        canvasHint.textContent = "before_img 中没有图片，可先运行 tools 截图";
        updateIdx();
        refreshBoxList();
        ctx.clearRect(0, 0, cv.width, cv.height);
      }
    } catch (e) {
      revokeBlobUrls();
      images = [];
      labelsByName = {};
      cur = -1;
      canvasHint.classList.remove("hidden");
      canvasHint.textContent =
        "无法连接标注服务：在项目根目录执行 python tools/fount/server.py 后刷新本页";
      updateIdx();
      refreshBoxList();
      ctx.clearRect(0, 0, cv.width, cv.height);
      setStatus(String(e.message || e), "err");
    }
  }

  function yoloTextForImage(name) {
    const boxes = labelsByName[name] || [];
    const lines = [];
    const meta = images.find((i) => i.name === name);
    const W = meta && meta.w ? meta.w : naturalW;
    const H = meta && meta.h ? meta.h : naturalH;
    if (!W || !H) return "";
    for (const b of boxes) {
      const x1 = Math.min(b.x1, b.x2);
      const x2 = Math.max(b.x1, b.x2);
      const y1 = Math.min(b.y1, b.y2);
      const y2 = Math.max(b.y1, b.y2);
      const bw = x2 - x1;
      const bh = y2 - y1;
      if (bw < 1 || bh < 1) continue;
      const xc = (x1 + x2) / 2 / W;
      const yc = (y1 + y2) / 2 / H;
      const nw = bw / W;
      const nh = bh / H;
      lines.push(
        `${b.cls} ${xc.toFixed(6)} ${yc.toFixed(6)} ${nw.toFixed(6)} ${nh.toFixed(6)}`
      );
    }
    return lines.join("\n");
  }

  function downloadText(filename, text) {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  document.getElementById("btnRefresh").addEventListener("click", () => {
    loadListFromServer();
  });

  document.getElementById("btnSave").addEventListener("click", async () => {
    const name = images[cur]?.name;
    if (!name) return;
    setStatus("保存中…", "");
    try {
      const r = await fetch("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json;charset=utf-8" },
        body: JSON.stringify({
          filename: name,
          labels: yoloTextForImage(name),
        }),
      });
      const raw = await r.text();
      if (!r.ok) throw new Error(raw || r.statusText);
      const stem = name.replace(/\.[^.]+$/, "");
      setStatus("已写入 img/" + name + " 与 labels/" + stem + ".txt", "ok");
    } catch (e) {
      setStatus("保存失败：" + (e.message || e), "err");
    }
  });

  inpImages.addEventListener("change", () => {
    const files = Array.from(inpImages.files || []);
    revokeBlobUrls();
    images = [];
    labelsByName = {};
    files.forEach((f) => {
      if (!/^image\//i.test(f.type)) return;
      images.push({
        name: f.name,
        file: f,
        url: URL.createObjectURL(f),
        server: false,
      });
    });
    cur = -1;
    if (images.length) loadImageAt(0);
    else {
      canvasHint.classList.remove("hidden");
      canvasHint.textContent = "请先选择图片或刷新 before_img";
      updateIdx();
      refreshBoxList();
      ctx.clearRect(0, 0, cv.width, cv.height);
    }
    inpImages.value = "";
  });

  function loadImageDims(im) {
    if (im.w && im.h) return Promise.resolve();
    return new Promise((res) => {
      const t = new Image();
      t.onload = () => {
        im.w = t.naturalWidth;
        im.h = t.naturalHeight;
        res();
      };
      t.onerror = () => res();
      t.src = im.url;
    });
  }

  inpLabels.addEventListener("change", async () => {
    const files = Array.from(inpLabels.files || []);
    for (const f of files) {
      const stem = f.name.replace(/\.txt$/i, "");
      const text = await f.text();
      const imgMatch = images.find((im) => im.name.replace(/\.[^.]+$/, "") === stem);
      if (!imgMatch) continue;
      await loadImageDims(imgMatch);
      const w0 = imgMatch.w;
      const h0 = imgMatch.h;
      if (!w0 || !h0) continue;
      const boxes = [];
      for (const line of text.split(/\r?\n/)) {
        const p = line.trim().split(/\s+/);
        if (p.length < 5) continue;
        const cls = parseInt(p[0], 10);
        const xc = parseFloat(p[1]);
        const yc = parseFloat(p[2]);
        const nw = parseFloat(p[3]);
        const nh = parseFloat(p[4]);
        const x1 = (xc - nw / 2) * w0;
        const y1 = (yc - nh / 2) * h0;
        const x2 = (xc + nw / 2) * w0;
        const y2 = (yc + nh / 2) * h0;
        boxes.push({ cls, x1, y1, x2, y2 });
      }
      labelsByName[imgMatch.name] = boxes;
    }
    if (cur >= 0) {
      refreshBoxList();
      draw();
    }
    inpLabels.value = "";
  });

  document.getElementById("btnPrev").addEventListener("click", () => {
    if (cur > 0) loadImageAt(cur - 1);
  });
  document.getElementById("btnNext").addEventListener("click", () => {
    if (cur < images.length - 1) loadImageAt(cur + 1);
  });

  document.getElementById("btnClear").addEventListener("click", () => {
    const name = images[cur]?.name;
    if (!name) return;
    labelsByName[name] = [];
    selectedBoxIndex = -1;
    refreshBoxList();
    draw();
  });

  document.getElementById("btnDlCurrent").addEventListener("click", () => {
    const name = images[cur]?.name;
    if (!name) return;
    const stem = name.replace(/\.[^.]+$/, "");
    downloadText(stem + ".txt", yoloTextForImage(name));
  });

  document.getElementById("btnDlZip").addEventListener("click", async () => {
    for (const im of images) await loadImageDims(im);
    if (typeof JSZip === "undefined") {
      images.forEach((im) => {
        const stem = im.name.replace(/\.[^.]+$/, "");
        downloadText(stem + ".txt", yoloTextForImage(im.name));
      });
      return;
    }
    const zip = new JSZip();
    for (const im of images) {
      const stem = im.name.replace(/\.[^.]+$/, "");
      zip.file(stem + ".txt", yoloTextForImage(im.name));
    }
    const blob = await zip.generateAsync({ type: "blob" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "yolo_labels.zip";
    a.click();
    URL.revokeObjectURL(a.href);
  });

  classSeg.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      classSeg.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentClass = parseInt(btn.dataset.cls, 10);
    });
  });

  function pointerToNat(ev) {
    const r = cv.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const y = ev.clientY - r.top;
    return clampNat(dispToNat(x, y).x, dispToNat(x, y).y);
  }

  cv.addEventListener("pointerdown", (ev) => {
    if (cur < 0 || !naturalW) return;
    ev.preventDefault();
    cv.setPointerCapture(ev.pointerId);
    drawing = true;
    startNat = pointerToNat(ev);
  });

  cv.addEventListener("pointermove", (ev) => {
    if (!drawing || !startNat) return;
    ev.preventDefault();
    const p = pointerToNat(ev);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(imgEl, 0, 0, cv.width, cv.height);
    const realBoxes = getBoxes();
    realBoxes.forEach((b, i) => {
      const x1 = b.x1 * scale;
      const y1 = b.y1 * scale;
      const x2 = b.x2 * scale;
      const y2 = b.y2 * scale;
      const col = colorForCls(b.cls);
      ctx.strokeStyle = i === selectedBoxIndex ? "#3d8bfd" : col;
      ctx.lineWidth = i === selectedBoxIndex ? 3 : 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
    const x1 = startNat.x * scale;
    const y1 = startNat.y * scale;
    const x2 = p.x * scale;
    const y2 = p.y * scale;
    ctx.strokeStyle = "#58a6ff";
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  });

  cv.addEventListener("pointerup", (ev) => {
    if (!drawing || !startNat) return;
    ev.preventDefault();
    try {
      cv.releasePointerCapture(ev.pointerId);
    } catch (_) {}
    drawing = false;
    const p = pointerToNat(ev);
    const x1 = Math.min(startNat.x, p.x);
    const x2 = Math.max(startNat.x, p.x);
    const y1 = Math.min(startNat.y, p.y);
    const y2 = Math.max(startNat.y, p.y);
    startNat = null;
    if (x2 - x1 < 4 || y2 - y1 < 4) {
      draw();
      return;
    }
    getBoxes().push({ cls: currentClass, x1, y1, x2, y2 });
    selectedBoxIndex = getBoxes().length - 1;
    refreshBoxList();
    draw();
  });

  cv.addEventListener("pointercancel", () => {
    drawing = false;
    startNat = null;
    draw();
  });

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input,textarea,button")) return;
    if (e.key === "1") classSeg.querySelector('[data-cls="0"]').click();
    if (e.key === "2") classSeg.querySelector('[data-cls="1"]').click();
    if (e.key === "3") classSeg.querySelector('[data-cls="2"]').click();
    if (e.key === "ArrowLeft" && cur > 0) loadImageAt(cur - 1);
    if (e.key === "ArrowRight" && cur < images.length - 1) loadImageAt(cur + 1);
    if (e.key === "Delete" || e.key === "Backspace") {
      if (selectedBoxIndex >= 0) {
        getBoxes().splice(selectedBoxIndex, 1);
        selectedBoxIndex = -1;
        refreshBoxList();
        draw();
      }
    }
  });

  window.addEventListener("resize", () => {
    if (cur >= 0 && imgEl) {
      layoutCanvas();
      draw();
    }
  });

  loadListFromServer();
})();
