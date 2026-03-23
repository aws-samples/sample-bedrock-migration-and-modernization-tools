import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3'

// S3 configuration for development proxy
const S3_BUCKET = process.env.VITE_S3_BUCKET || 'your-data-bucket-name'
const S3_REGION = 'us-east-1'

// Create S3 client (uses default credential chain - AWS CLI credentials)
const s3Client = new S3Client({ region: S3_REGION })

/**
 * Custom Vite plugin to proxy S3 requests during development.
 * This allows fetching private S3 data using local AWS credentials.
 * In production, CloudFront serves data directly from /data/* path.
 */
function s3ProxyPlugin() {
  return {
    name: 's3-proxy',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        // Only handle /s3-data/* requests
        if (!req.url?.startsWith('/s3-data/')) {
          return next()
        }

        const s3Key = req.url.replace('/s3-data/', '')

        try {
          const command = new GetObjectCommand({
            Bucket: S3_BUCKET,
            Key: s3Key,
          })

          const response = await s3Client.send(command)
          const body = await response.Body.transformToString()

          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Access-Control-Allow-Origin', '*')
          res.end(body)
        } catch (error) {
          console.error(`[S3 Proxy] Failed to fetch ${s3Key}:`, error.message)
          res.statusCode = 500
          res.end(JSON.stringify({ error: `Failed to fetch from S3: ${error.message}` }))
        }
      })
    },
  }
}

export default defineConfig(({ mode }) => {
  const isDev = mode === 'development'

  if (isDev) {
    console.log(`\n[Vite] Development mode - S3 proxy enabled for /s3-data/* requests\n`)
  }

  return {
    plugins: [
      react(),
      // S3 proxy only needed in development
      ...(isDev ? [s3ProxyPlugin()] : []),
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    build: {
      // Optimize production build
      sourcemap: false,
      minify: 'esbuild',
    },
  }
})
