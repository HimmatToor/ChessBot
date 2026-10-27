import type { NextConfig } from "next";

// Cross-origin isolation, so the engine can run on several threads (see public/_headers).
const isolationHeaders = [
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
];

const nextConfig: NextConfig = {
  // Build to plain static files in out/ so it can be hosted for free on
  // Cloudflare Pages. The whole app (including the bot) runs in the browser.
  output: "export",
  // The Next.js image optimiser needs a server, which a static export doesn't have.
  images: { unoptimized: true },
  // A static export can't set headers (Cloudflare reads public/_headers instead),
  // so this only applies to `next dev`, keeping local behaviour the same as production.
  ...(process.env.NODE_ENV === "development" && {
    async headers() {
      return [{ source: "/:path*", headers: isolationHeaders }];
    },
  }),
};

export default nextConfig;
