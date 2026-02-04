const { URL } = require("url");

async function readUrlParam(req) {
  try {
    const base = `http://${req.headers.host || "localhost"}`;
    const parsed = new URL(req.url, base);
    const raw = parsed.searchParams.get("url");
    return raw ? decodeURIComponent(raw) : "";
  } catch {
    return "";
  }
}

async function fetchImage(url, headers) {
  const response = await fetch(url, {
    headers,
    redirect: "follow",
  });
  if (!response.ok) {
    const err = new Error(`Upstream error: ${response.status}`);
    err.status = response.status;
    throw err;
  }
  return response;
}

function buildCandidates(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return [];
  }

  const candidates = [];
  candidates.push(parsed.toString());

  const noQuery = new URL(parsed.toString());
  noQuery.search = "";
  candidates.push(noQuery.toString());

  if (parsed.hostname === "media.discordapp.net") {
    const cdn = new URL(parsed.toString());
    cdn.hostname = "cdn.discordapp.com";
    candidates.push(cdn.toString());
    const cdnNoQuery = new URL(cdn.toString());
    cdnNoQuery.search = "";
    candidates.push(cdnNoQuery.toString());
  }

  return Array.from(new Set(candidates));
}

module.exports = async (req, res) => {
  if (req.method !== "GET") {
    res.statusCode = 405;
    res.setHeader("Content-Type", "text/plain");
    res.end("Method not allowed");
    return;
  }

  const targetUrl = await readUrlParam(req);
  if (!targetUrl || !/^https?:\/\//i.test(targetUrl)) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "text/plain");
    res.end("Missing or invalid url parameter");
    return;
  }

  const headers = {
    "User-Agent": "OperatorAssist/1.0",
    Accept: "image/avif,image/webp,image/*,*/*;q=0.8",
    Referer: "https://discord.com/",
    Origin: "https://discord.com",
  };

  const candidates = buildCandidates(targetUrl);
  if (!candidates.length) {
    res.statusCode = 400;
    res.setHeader("Content-Type", "text/plain");
    res.end("Invalid image url");
    return;
  }

  let lastError = null;
  for (const candidate of candidates) {
    try {
      const response = await fetchImage(candidate, headers);
      const contentType = response.headers.get("content-type") || "application/octet-stream";
      res.statusCode = 200;
      res.setHeader("Content-Type", contentType);
      res.setHeader("Cache-Control", "no-store");

      const buffer = Buffer.from(await response.arrayBuffer());
      res.end(buffer);
      return;
    } catch (err) {
      lastError = err;
    }
  }

  res.statusCode = lastError?.status || 502;
  res.setHeader("Content-Type", "text/plain");
  res.end("Failed to fetch image");
};
