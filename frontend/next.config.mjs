/** @type {import('next').NextConfig} */
// Built as static files and served by the API container (pmv_turismo.server).
// NEXT_PUBLIC_BASE_PATH is the path the app lives under behind the gateway
// (e.g. /tourism); empty for local development.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH || ''

const nextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath,
  images: {
    // No image optimisation server in a static export: Wikimedia serves the sizes.
    unoptimized: true,
    remotePatterns: [
      { protocol: 'https', hostname: 'commons.wikimedia.org' },
      { protocol: 'https', hostname: 'upload.wikimedia.org' },
      { protocol: 'https', hostname: 'thumb.wikimedia.org' },
    ],
  },
}

export default nextConfig
