import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Build to plain static files in out/ so it can be hosted for free on
  // Cloudflare Pages. The whole app (including the bot) runs in the browser.
  output: "export",
  // The Next.js image optimiser needs a server, which a static export doesn't have.
  images: { unoptimized: true },
};

export default nextConfig;
