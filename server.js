const { createServer } = require('http')
const { parse } = require('url')
const next = require('next')
const WebSocket = require('ws')
// const fetch = require('node-fetch') // Import node-fetch to make HTTP requests
const KEEP_ALIVE_INTERVAL = 30000 // 30 seconds ping
const CLIENT_TIMEOUT = 60 * 60 * 1000 // 60 minutes

const dev = process.env.NODE_ENV !== 'production'
const app = next({ dev: true })
const handle = app.getRequestHandler()

app.prepare().then(() => {
  const server = createServer((req, res) => {
    const parsedUrl = parse(req.url, true)
    handle(req, res, parsedUrl)
  })

  const wss = new WebSocket.Server({ 
    noServer: true // Ensures that WebSocket is handled manually
  })

  // Handle WebSocket upgrades
  server.on('upgrade', (req, socket, head) => {
    const parsedUrl = parse(req.url, true)

    // Check if the request is for the /socket path
    if (parsedUrl.pathname === '/socket') {
      wss.handleUpgrade(req, socket, head, (ws) => {
        wss.emit('connection', ws, req)
      })
    } else {
      socket.destroy() // Reject the connection if it's not for /socket
    }
  })

  wss.on('connection', (ws) => {
    console.log('Client connected via WebSocket')

     // Keep-alive ping every 30s
     const keepAlive = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.ping()
        }
      }, KEEP_ALIVE_INTERVAL)

      // Optional: close connection after 60 min
    const autoClose = setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close(1000, 'Session timed out after 60 minutes')
        }
      }, CLIENT_TIMEOUT)

    ws.on('message', async (message) => {
      console.log(`Received: ${message}`)
      const parsedMessage = JSON.parse(message);
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(parsedMessage),
      })
      const responseBody = await response.json()
      ws.send(JSON.stringify(responseBody))
    })

    ws.on('close', () => {
        console.log('Client disconnected')
        clearInterval(keepAlive)
        clearTimeout(autoClose)
      })
  })

  server.listen(3000, () => {
    console.log('> Ready on http://localhost:3000')
  })
})
