import type { NextConfig } from "next";

/** Where Next.js proxies browser requests to `/api/*` when the UI uses same-origin paths. */
const backendProxy =
  process.env.BACKEND_PROXY_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendProxy}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
