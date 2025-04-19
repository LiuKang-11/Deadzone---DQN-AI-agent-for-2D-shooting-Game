# ai_server.py

import socket
import torch
import torch.nn as nn

# 1) 定义网络结构，须与训练时完全一致
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    HOST, PORT = '127.0.0.1', 5001
    state_size, action_size = 11, 12

    model = QNetwork(state_size, action_size)
    #best result right now
    #model.load_state_dict(torch.load('models2/player_winner_ep17.pth', map_location='cpu'))77 3_263 3_939 2_822/823/824/846/847
    #model.load_state_dict(torch.load('models/player_winner_ep59.pth', map_location='cpu')) 

    model.load_state_dict(torch.load('models3/player_winner_ep1200.pth', map_location='cpu'))
    #model.load_state_dict(torch.load('models3/opponent_winner_ep661.pth', map_location='cpu'))

    model.eval()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    print(f"[AI] Listening on {HOST}:{PORT} ...")

    conn, addr = srv.accept()
    print(f"[AI] Connected by {addr}")
    buf = b''

    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                print("[AI] Connection closed by client")
                break
            buf += chunk

            while True:
                if len(buf) < 4: break
                total_size = int.from_bytes(buf[0:4], 'big')
                if len(buf) < 4 + total_size: break
                packet = buf[4:4+total_size]
                buf = buf[4+total_size:]

                if len(packet) < 4:
                    print("[AI] Malformed packet (too small)")
                    continue
                str_len = int.from_bytes(packet[0:4], 'big')
                if len(packet) < 4 + str_len:
                    print("[AI] Incomplete string payload")
                    continue

                payload = packet[4:4+str_len]
                msg = payload.decode('utf-8', errors='ignore')
                print(f"[AI] Raw msg: {repr(msg)}")

                if msg.startswith("P0:") or msg.startswith("P1:"):
                    who = msg[:2]
                    msg = msg[3:]
                else:
                    print("[AI] Missing player prefix")
                    conn.sendall(b"-1")
                    continue

                try:
                    floats = [float(x) for x in msg.split(',') if x]
                except ValueError as e:
                    print("[AI] parse error:", e)
                    conn.sendall(b"-1")
                    continue

                print(f"[AI] [{who}] Parsed floats:", floats)
                state = torch.tensor(floats, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    qvals = model(state)
                    action = int(torch.argmax(qvals, dim=1).item())
                print(f"[AI] [{who}] Predicted action: {action}")

                # 正確送法（符合 SFML 封包格式）
                action_str = str(action).encode()
                inner = len(action_str).to_bytes(4, 'big') + action_str
                outer = len(inner).to_bytes(4, 'big') + inner
                conn.sendall(outer)
    finally:
        conn.close()
        srv.close()

if __name__ == '__main__':
    main()

