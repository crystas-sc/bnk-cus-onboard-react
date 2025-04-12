// src/app/api/socket/route.js
import WebSocket from "ws"

let wss

export function setupWebSocket(server) {
  if (!wss) {
    wss = new WebSocket.Server({ server })

    wss.on('connection', (ws) => {
      console.log('WebSocket connected')

      ws.on('message', (message) => {
        console.log(`Received: ${message}`)
        ws.send(`Echo: ${message}`)
      })

      ws.on('close', () => {
        console.log('WebSocket closed')
      })
    })

    server.on('upgrade', (req, socket, head) => {
      // Only upgrade to WebSocket if the path is "/api/socket"
      if (req.url === '/api/socket') {
        wss.handleUpgrade(req, socket, head, (ws) => {
          wss.emit('connection', ws, req)
        })
      } else {
        socket.destroy()
      }
    })
  }

  return wss
}
