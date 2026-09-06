/*
 VexelLibFaceDetection - Real-time Webcam C++ Demo
 Copyright (c) 2026, Vexel Innovations. All rights reserved.
*/

#include "vexelfacedetect.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    int camera_id = 0;
    if (argc > 1) {
        camera_id = std::stoi(argv[1]);
    }

    std::cout << "==============================================" << std::endl;
    std::cout << " VexelLibFaceDetection | Real-Time Webcam Demo" << std::endl;
    std::cout << " Powered by Vexel Innovations C++20 Engine   " << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Opening camera ID: " << camera_id << "..." << std::endl;
    std::cout << "Press 'ESC' or 'q' in the window to exit." << std::endl;

    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open webcam with ID " << camera_id << std::endl;
        return -1;
    }

    // Set resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    vexelfacedetect::FaceDetector detector;
    vexelfacedetect::DetectionOptions options;
    options.confidence_threshold = 0.5f;

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        auto start = std::chrono::high_resolution_clock::now();

        // OpenCV BGR frame data
        std::span<const uint8_t> img_span(frame.data, frame.total() * frame.elemSize());

        auto faces = detector.detect(img_span, frame.cols, frame.rows, static_cast<int>(frame.step), options);

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        double fps = 1000.0 / (ms + 0.0001);

        // Draw faces & landmarks
        for (const auto& face : faces) {
            // Draw Bounding Box (Vexel Cyan/Teal #00D2FF)
            cv::Rect rect(face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height);
            cv::rectangle(frame, rect, cv::Scalar(255, 210, 0), 2);

            // Score tag
            std::string text = cv::format("Vexel: %.2f", face.score);
            cv::putText(frame, text, cv::Point(face.bbox.x, face.bbox.y - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 210, 0), 2);

            // Draw 5 Facial Landmarks (Red dots for eyes/nose/mouth)
            for (int i = 0; i < 5; ++i) {
                cv::circle(frame, cv::Point(face.landmarks[i].x, face.landmarks[i].y),
                           3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Overlay stats
        std::string stats = cv::format("FPS: %.1f | Latency: %.2f ms | Faces: %zu", fps, ms, faces.size());
        cv::putText(frame, stats, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

        cv::imshow("VexelLibFaceDetection - Live Camera", frame);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
