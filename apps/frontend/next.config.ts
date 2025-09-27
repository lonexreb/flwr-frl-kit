import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  outputFileTracingRoot: '/app/apps/frontend',
  eslint: {
    ignoreDuringBuilds: true
  }
};

export default nextConfig;
