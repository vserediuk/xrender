//
// Created by Vasyl on 29.12.2024.
//

#ifndef CAMERA_H
#define CAMERA_H

#include <unordered_map>
#include <glm/glm.hpp>
#include <SDL2/SDL_events.h>


class Camera {
public:
    Camera();
    ~Camera();

    void update(float deltaTime);
    void processInput(const SDL_Event& event);
    void processMouseMotion(const SDL_Event& event);

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::vec3 getPosition() const;
    glm::vec3 getFront() const;
    glm::vec3 getUp() const;

private:
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec3 worldUp;

    float yaw;
    float pitch;
    float movementSpeed;
    float mouseSensitivity;
    float zoom;

    std::unordered_map<SDL_Keycode, bool> keys;

    void updateCameraVectors();
    float lastX, lastY;
    bool firstMouse;
};

#endif //CAMERA_H
