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

  try {
    const response = await fetch(targetUrl, {
      headers: {
        "User-Agent": "OperatorAssist/1.0",
      },
    });
    if (!response.ok) {
      res.statusCode = response.status;
      res.setHeader("Content-Type", "text/plain");
      res.end(`Upstream error: ${response.status}`);
      return;
    }

    const contentType = response.headers.get("content-type") || "application/octet-stream";
    res.statusCode = 200;
    res.setHeader("Content-Type", contentType);
    res.setHeader("Cache-Control", "no-store");

    const buffer = Buffer.from(await response.arrayBuffer());
    res.end(buffer);
  } catch (err) {
    res.statusCode = 502;
    res.setHeader("Content-Type", "text/plain");
    res.end("Failed to fetch image");
  }
};
