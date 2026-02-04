const params = new URLSearchParams(window.location.search);
const imgUrl = params.get("img");
const left = Number(params.get("left") || 0);
const top = Number(params.get("top") || 0);
const session = params.get("session") || "";
const step = params.get("step") || "";

const imageWrap = document.getElementById("image-wrap");
const img = document.getElementById("shot");
const crosshair = document.getElementById("crosshair");
const coordsEl = document.getElementById("coords");
const absCoordsEl = document.getElementById("abs-coords");
const confirmBtn = document.getElementById("confirm");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("image-error");

let selection = null;
let retried = false;

function loadImage() {
  if (!imgUrl) {
    errorEl.classList.remove("hidden");
    return;
  }
  statusEl.textContent = "Loading image...";
  img.referrerPolicy = "no-referrer";
  img.crossOrigin = "anonymous";
  const proxiedUrl = `/api/image?url=${encodeURIComponent(imgUrl)}`;
  img.src = proxiedUrl;
}

loadImage();

img.addEventListener("error", () => {
  if (imgUrl && !retried && imgUrl.includes("?")) {
    retried = true;
    const baseUrl = imgUrl.split("?")[0];
    const proxiedUrl = `/api/image?url=${encodeURIComponent(baseUrl)}`;
    img.src = proxiedUrl;
    return;
  }
  if (imgUrl && img.src.includes("/api/image")) {
    // Proxy failed, try direct CDN.
    img.src = imgUrl;
    statusEl.textContent = "Proxy failed, trying direct image...";
    return;
  }
  if (imgUrl && !img.src.endsWith(imgUrl) && imgUrl.includes("?")) {
    // Direct with query failed, try without query.
    img.src = imgUrl.split("?")[0];
    return;
  }
  errorEl.classList.remove("hidden");
  statusEl.textContent = "Failed to load image. The CDN link may be expired or blocked.";
});

img.addEventListener("load", () => {
  errorEl.classList.add("hidden");
  statusEl.textContent = "Click on the image to set a target.";
});

imageWrap.addEventListener("click", (event) => {
  if (!img.naturalWidth || !img.naturalHeight) {
    return;
  }
  const rect = img.getBoundingClientRect();
  const x = Math.max(0, Math.min(rect.width, event.clientX - rect.left));
  const y = Math.max(0, Math.min(rect.height, event.clientY - rect.top));

  const scaleX = img.naturalWidth / rect.width;
  const scaleY = img.naturalHeight / rect.height;
  const imgX = Math.round(x * scaleX);
  const imgY = Math.round(y * scaleY);

  selection = { x: imgX, y: imgY };

  crosshair.style.left = `${x}px`;
  crosshair.style.top = `${y}px`;
  crosshair.classList.remove("hidden");

  coordsEl.textContent = `${imgX}, ${imgY}`;
  absCoordsEl.textContent = `${imgX + left}, ${imgY + top}`;
  confirmBtn.disabled = false;
  statusEl.textContent = "Ready to send click.";
});

confirmBtn.addEventListener("click", async () => {
  if (!selection) {
    return;
  }
  confirmBtn.disabled = true;
  statusEl.textContent = "Sending click...";

  try {
    const res = await fetch("/api/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        x: selection.x,
        y: selection.y,
        left,
        top,
        session,
        step,
        img: imgUrl,
      }),
    });

    if (!res.ok) {
      throw new Error(`Request failed (${res.status})`);
    }
    const data = await res.json();
    statusEl.textContent = `Click sent: ${data.absX}, ${data.absY}`;
  } catch (err) {
    statusEl.textContent = `Failed to send click: ${err.message}`;
  } finally {
    confirmBtn.disabled = false;
  }
});
