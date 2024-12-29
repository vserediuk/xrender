//
// Created by Vasyl on 29.12.2024.
//

#include "camera.h"
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

Camera::Camera()
    : position(glm::vec3(0.0f, 0.0f, 3.0f)),  // Camera starts at (0, 0, 3)
      front(glm::vec3(0.0f, 0.0f, -1.0f)),
      up(glm::vec3(0.0f, 1.0f, 0.0f)),
      right(glm::vec3(1.0f, 0.0f, 0.0f)),
      worldUp(glm::vec3(0.0f, 1.0f, 0.0f)),
      yaw(-90.0f),
      pitch(0.0f),
      movementSpeed(2.5f),
      mouseSensitivity(0.1f),
      zoom(45.0f) {
    keys[SDLK_w] = false;
    keys[SDLK_s] = false;
    keys[SDLK_a] = false;
    keys[SDLK_d] = false;
    updateCameraVectors();
}

Camera::~Camera() {}

void Camera::update(float deltaTime) {
    float velocity = movementSpeed * deltaTime;

    if (keys[SDLK_w])
        position += front * velocity;
    if (keys[SDLK_s])
        position -= front * velocity;
    if (keys[SDLK_a])
        position -= right * velocity;
    if (keys[SDLK_d])
        position += right * velocity;

    updateCameraVectors();
}


void Camera::processInput(const SDL_Event& event) {
    if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
        bool state = event.type == SDL_KEYDOWN;

        if (event.key.keysym.sym == SDLK_w)
            keys[SDLK_w] = state;
        if (event.key.keysym.sym == SDLK_s)
            keys[SDLK_s] = state;
        if (event.key.keysym.sym == SDLK_a)
            keys[SDLK_a] = state;
        if (event.key.keysym.sym == SDLK_d)
            keys[SDLK_d] = state;
    }
}



void Camera::processMouseMotion(const SDL_Event &event) {
    if (firstMouse) {
        lastX = event.motion.x;
        lastY = event.motion.y;
        firstMouse = false;
    }

    float xOffset = event.motion.x - lastX;
    float yOffset = lastY - event.motion.y;
    lastX = event.motion.x;
    lastY = event.motion.y;

    xOffset *= mouseSensitivity;
    yOffset *= mouseSensitivity;

    yaw += xOffset;
    pitch += yOffset;

    if (pitch > 89.0f)
        pitch = 89.0f;
    if (pitch < -89.0f)
        pitch = -89.0f;

    updateCameraVectors();
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + front, up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    return glm::perspective(glm::radians(zoom), 800.0f / 600.0f, 0.1f, 100.0f); // Replace with actual window dimensions
}

glm::vec3 Camera::getPosition() const {
    return position;
}

glm::vec3 Camera::getFront() const {
    return front;
}

glm::vec3 Camera::getUp() const {
    return up;
}

void Camera::updateCameraVectors() {
    glm::vec3 newFront;
    newFront.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    newFront.y = sin(glm::radians(pitch));
    newFront.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(newFront);

    right = glm::normalize(glm::cross(front, worldUp));  // Right vector
    up = glm::normalize(glm::cross(right, front));  // Up vector
}

