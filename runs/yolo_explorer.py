import os
import cv2
import numpy as np
import vizdoom as vzd
import time

OUTPUT_DIR = "yolo_dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

game = vzd.DoomGame()
game.load_config(vzd.scenarios_path + "/deathmatch.cfg")
game.set_window_visible(True)
game.set_labels_buffer_enabled(True)
game.set_objects_info_enabled(True)

game.add_game_args("+sv_cheats 1 +god 1 +timelimit 999")

game.init()
num_frames = 5000
n_buttons = game.get_available_buttons_size()

game.new_episode()
frame_id = 0

ENEMY_CLASSES = {
    "DoomPlayer",
    "Zombieman",
    "ShotgunGuy",
    "ChaingunGuy",
    "Demon",
    "Spectre",
    "Imp",
    "Cacodemon",
    "BaronOfHell",
    "HellKnight"
}

while frame_id < num_frames:
    time.sleep(0.01)
    game.send_game_command("give health 1")

    # Zufällige Aktion
    action = np.zeros(n_buttons, dtype=int)
    
    if np.random.rand() < 0.6:
        action[game.get_available_buttons().index(vzd.Button.MOVE_FORWARD)] = 1
    elif np.random.rand() < 0.3:
        action[game.get_available_buttons().index(vzd.Button.MOVE_BACKWARD)] = 1

    if np.random.rand() < 0.5:
        action[game.get_available_buttons().index(vzd.Button.TURN_LEFT)] = 1
    #else:
        #action[game.get_available_buttons().index(vzd.Button.TURN_RIGHT)] = 1

    if np.random.rand() < 0.2:
        action[game.get_available_buttons().index(vzd.Button.ATTACK)] = 1

    game.make_action(action.tolist())

    state = game.get_state()

    if game.is_episode_finished():
        game.new_episode()
        continue

    if state is not None:
        frame = state.screen_buffer.transpose(1, 2, 0).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_file = os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.png")
        txt_file = os.path.join(OUTPUT_DIR, f"frame_{frame_id:05d}.txt")

        cv2.imwrite(frame_file, frame)

        labels = state.labels
        with open(txt_file, "w") as f:
            if labels:
                for label in labels:
                    class_id = label.object_id
                    class_name = label.object_name

                    if class_name not in ENEMY_CLASSES:
                        continue

                    x, y, w, h = label.x, label.y, label.width, label.height
                    xc = (x + w / 2) / frame.shape[1]
                    yc = (y + h / 2) / frame.shape[0]
                    wn = w / frame.shape[1]
                    hn = h / frame.shape[0]
                    if w < 10 or h < 10:
                        continue
                    f.write(f"{class_name} {class_id} {xc} {yc} {wn} {hn}\n")
                    #f.write(f"{class_id} {xc} {yc} {wn} {hn}\n")
                    
            else:
                f.write("")  # YOLO erwartet Textdatei, auch wenn leer

        frame_id += 1

game.close()