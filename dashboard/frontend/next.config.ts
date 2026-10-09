import type { NextConfig } from 'next';
const nextConfig: NextConfig = {
  devIndicators: false,
  async rewrites() {
    return [{ source: '/api/:path*', destination: `${process.env.TGDETECT_API_ORIGIN || 'http://127.0.0.1:8000'}/api/:path*` }];
  },
};
export default nextConfig;
