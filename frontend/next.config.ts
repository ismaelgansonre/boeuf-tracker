import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const backend = process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:5000";
    return [
      { source: "/api/:path*", destination: `${backend}/api/:path*` },
      { source: "/video_feed", destination: `${backend}/video_feed` },
    ];
  },
};

export default nextConfig;
