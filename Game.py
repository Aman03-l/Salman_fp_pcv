import cv2
import numpy as np
import mediapipe as mp
import time
import random
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def overlay_sprite(background, sprite, x, y):
    h, w, _ = sprite.shape
    bh, bw, _ = background.shape

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, bw), min(y + h, bh)

    sw1, sh1 = x1 - x, y1 - y
    sw2, sh2 = sw1 + (x2 - x1), sh1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return background

    roi = background[y1:y2, x1:x2].astype(float)
    sub_sprite = sprite[sh1:sh2, sw1:sw2].astype(float)

    alpha = sub_sprite[:, :, 3:4] / 255.0
    blended = (sub_sprite[:, :, :3] * alpha) + (roi * (1 - alpha))

    background[y1:y2, x1:x2] = blended.astype(np.uint8)
    return background


def player_sprite(size=60):
    sprite = np.zeros((size, size, 4), dtype=np.uint8)

    center_x = size // 2

    pts_body = np.array([
        [center_x, 5],
        [center_x - 15, size - 12],
        [center_x, size - 22],
        [center_x + 15, size - 12]
    ], np.int32)

    cv2.fillPoly(sprite, [pts_body], (255, 180, 50, 255))

    pts_left = np.array([
        [center_x - 8, 30],
        [5, size - 15],
        [center_x - 4, size - 25]
    ], np.int32)

    pts_right = np.array([
        [center_x + 8, 30],
        [size - 5, size - 15],
        [center_x + 4, size - 25]
    ], np.int32)

    cv2.fillPoly(sprite, [pts_left], (255, 80, 80, 255))
    cv2.fillPoly(sprite, [pts_right], (255, 80, 80, 255))
    cv2.circle(sprite, (center_x, 25), 6, (255, 255, 255, 255), -1)
    cv2.polylines(sprite, [pts_body], True, (30, 30, 30, 255), 2)

    return sprite

def draw_text_center(frame, text, y, scale=1.0, color=(255, 255, 255), thickness=2):
    h, w, _ = frame.shape
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    text_w = text_size[0]

    x = (w - text_w) // 2

    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def count_fingers(hand_landmarks, handedness_label):
    fingers = 0
    tips = [8, 12, 16, 20]

    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers += 1
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    if handedness_label == "Right":
        if thumb_tip.x < thumb_ip.x:
            fingers += 1
    else:
        if thumb_tip.x > thumb_ip.x:
            fingers += 1

    return fingers


def reset_enemy(w_frame, h_frame, score):
    level = 1 + score // 10

    enemy = {
        "x": w_frame + random.randint(0, 200),
        "y": random.randint(60, h_frame - 60),
        "radius": 24 + min(level * 2, 18),
        "speed": 5 + min(level, 10),
        "hp": 1 + (score // 10) * 2,
        "max_hp": 1 + (score // 10) * 2,
        "color": (0, 0, 255)
    }

    return enemy

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

frame = cv2.flip(frame, 1)
h_frame, w_frame, _ = frame.shape

STATE_START = "start"
STATE_PLAYING = "playing"
STATE_GAME_OVER = "game_over"

game_state = STATE_START

# Player
player = {
    "x": 120,
    "y": h_frame // 2,
    "target_x": 120,
    "target_y": h_frame // 2,
    "radius": 28,
    "hp": 5,
    "max_hp": 5,
    "invincible_until": 0
}

enemy = reset_enemy(w_frame, h_frame, 0)

bullets = []
enemy_bullets = []

score = 0
level = 1
last_shot_time = 0
last_enemy_shot_time = 0
last_damage_time = 0
fire_rate = 0.25
enemy_fire_rate = 1.2
start_time = time.time()
smoothness = 0.35

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    
    h_frame, w_frame, _ = frame.shape

    frame = cv2.convertScaleAbs(frame, alpha=0.55, beta=-10)
    
    current_time = time.time()
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)

    num_fingers = 0
    
    hand_detected = False

    if results.multi_hand_landmarks:
        for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_detected = True

            handedness_label = "Right"

            if results.multi_handedness:
                handedness_label = results.multi_handedness[index].classification[0].label

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            player["target_x"] = int(wrist.x * w_frame)
            player["target_y"] = int(wrist.y * h_frame)

            num_fingers = count_fingers(hand_landmarks, handedness_label)

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    player["x"] += int((player["target_x"] - player["x"]) * smoothness)
    player["y"] += int((player["target_y"] - player["y"]) * smoothness)

    player["x"] = max(30, min(w_frame - 30, player["x"]))
    player["y"] = max(30, min(h_frame - 30, player["y"]))

    is_slowmo = num_fingers in [4, 5]
    is_shooting = num_fingers in [3, 5]

    time_scale = 0.35 if is_slowmo else 1.0

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    if key == ord(" "):
        if game_state in [STATE_START, STATE_GAME_OVER]:
            score = 0
            level = 1

            player["x"] = 120
            player["y"] = h_frame // 2
            player["target_x"] = 120
            player["target_y"] = h_frame // 2
            player["hp"] = player["max_hp"]
            player["invincible_until"] = 0

            bullets.clear()
            enemy_bullets.clear()

            enemy = reset_enemy(w_frame, h_frame, score)

            start_time = current_time
            last_shot_time = 0
            last_enemy_shot_time = 0
            last_damage_time = 0

            game_state = STATE_PLAYING

    if game_state == STATE_START:
        draw_text_center(frame, "Tekan SPACE untuk mulai", h_frame // 2 + 70, 0.8, (0, 255, 0), 2)
        cv2.imshow("Platypus", frame)
        continue

    if game_state == STATE_PLAYING:
        level = 1 + score // 10

        if is_shooting and current_time - last_shot_time > fire_rate:
            bullets.append({
                "x": player["x"] + 32,
                "y": player["y"],
                "speed": 12,
                "radius": 5,
                "damage": 1
            })
            last_shot_time = current_time

        enemy["x"] -= int(enemy["speed"] * time_scale)

        if enemy["x"] < -enemy["radius"]:
            enemy = reset_enemy(w_frame, h_frame, score)
            player["hp"] -= 1
            player["invincible_until"] = current_time + 1.0

            if player["hp"] <= 0:
                game_state = STATE_GAME_OVER

        if current_time - last_enemy_shot_time > enemy_fire_rate:
            dx = player["x"] - enemy["x"]
            dy = player["y"] - enemy["y"]
            length = math.sqrt(dx * dx + dy * dy)

            if length != 0:
                dx /= length
                dy /= length

            enemy_bullets.append({
                "x": enemy["x"],
                "y": enemy["y"],
                "vx": dx * (5 + min(level, 6)),
                "vy": dy * (5 + min(level, 6)),
                "radius": 6
            })

            last_enemy_shot_time = current_time

        for bullet in bullets[:]:
            bullet["x"] += int(bullet["speed"] * time_scale)

            if bullet["x"] > w_frame + 50:
                bullets.remove(bullet)
                continue

            if distance(bullet["x"], bullet["y"], enemy["x"], enemy["y"]) < bullet["radius"] + enemy["radius"]:
                enemy["hp"] -= bullet["damage"]

                if bullet in bullets:
                    bullets.remove(bullet)

                if enemy["hp"] <= 0:
                    score += 1
                    enemy = reset_enemy(w_frame, h_frame, score)

        for eb in enemy_bullets[:]:
            eb["x"] += eb["vx"] * time_scale
            eb["y"] += eb["vy"] * time_scale

            if eb["x"] < -50 or eb["x"] > w_frame + 50 or eb["y"] < -50 or eb["y"] > h_frame + 50:
                enemy_bullets.remove(eb)
                continue

            hit_player = distance(eb["x"], eb["y"], player["x"], player["y"]) < eb["radius"] + player["radius"]

            if hit_player and current_time > player["invincible_until"]:
                player["hp"] -= 1
                player["invincible_until"] = current_time + 1.0

                if eb in enemy_bullets:
                    enemy_bullets.remove(eb)

                if player["hp"] <= 0:
                    game_state = STATE_GAME_OVER

        hit_enemy = distance(player["x"], player["y"], enemy["x"], enemy["y"]) < player["radius"] + enemy["radius"]

        if hit_enemy and current_time > player["invincible_until"]:
            player["hp"] -= 1
            player["invincible_until"] = current_time + 1.0

            enemy["hp"] -= 1

            if enemy["hp"] <= 0:
                score += 1
                enemy = reset_enemy(w_frame, h_frame, score)

            if player["hp"] <= 0:
                game_state = STATE_GAME_OVER

    if is_slowmo and game_state == STATE_PLAYING:
        cv2.rectangle(frame, (0, 0), (w_frame - 1, h_frame - 1), (255, 255, 0), 5)
        cv2.putText(
            frame,
            "SLOW MOTION",
            (w_frame - 190, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

    for bullet in bullets:
        cv2.circle(
            frame,
            (int(bullet["x"]), int(bullet["y"])),
            bullet["radius"],
            (0, 255, 255),
            -1
        )

        cv2.circle(
            frame,
            (int(bullet["x"]) - 8, int(bullet["y"])),
            3,
            (0, 120, 255),
            -1
        )

    for eb in enemy_bullets:
        cv2.circle(
            frame,
            (int(eb["x"]), int(eb["y"])),
            eb["radius"],
            (255, 0, 255),
            -1
        )

    enemy_color_strength = max(80, 255 - level * 15)
    enemy_color = (0, 0, enemy_color_strength)

    cv2.circle(
        frame,
        (int(enemy["x"]), int(enemy["y"])),
        enemy["radius"],
        enemy_color,
        -1
    )

    cv2.circle(
        frame,
        (int(enemy["x"]), int(enemy["y"])),
        enemy["radius"],
        (255, 255, 255),
        2
    )

    blink = current_time < player["invincible_until"] and int(current_time * 10) % 2 == 0

    if not blink:
        current_sprite = player_sprite()
        frame = overlay_sprite(
            frame,
            current_sprite,
            int(player["x"] - current_sprite.shape[1] // 2),
            int(player["y"] - current_sprite.shape[0] // 2)
        )

    # UI panel
    cv2.rectangle(frame, (0, 0), (w_frame, 58), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"Score: {score}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Level: {level}",
        (150, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Fingers: {num_fingers}",
        (285, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (180, 255, 180),
        2
    )

    if game_state == STATE_GAME_OVER:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w_frame, h_frame), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        draw_text_center(frame, "GAME OVER", h_frame // 2 - 70, 1.6, (0, 0, 255), 4)
        draw_text_center(frame, f"Final Score: {score}", h_frame // 2 - 10, 0.9, (255, 255, 255), 2)
        draw_text_center(frame, "Tekan SPACE untuk restart", h_frame // 2 + 90, 0.8, (0, 255, 0), 2)
        draw_text_center(frame, "Tekan Q untuk keluar", h_frame // 2 + 130, 0.65, (180, 180, 180), 1)

    cv2.imshow("Platypus", frame)

cap.release()
cv2.destroyAllWindows()
