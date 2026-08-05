//
//  Pipeline.swift
//  Orchestration : FrameSource → Detector (ANE) → état observable pour l'UI.
//
//  @Observable : SwiftUI se met à jour tout seul quand `detections` / `count`
//  changent. Le travail ANE se fait hors main thread ; on repasse sur le
//  main uniquement pour publier le résultat.
//

import Observation
import CoreVideo
import QuartzCore

@Observable
final class Pipeline {
    // État publié vers l'UI
    private(set) var detections: [Detection] = []
    private(set) var count: Int = 0
    private(set) var fps: Double = 0
    private(set) var status: String = "Prêt"

    private var source: FrameSource?
    private var detector: Detector?
    private let work = DispatchQueue(label: "boeuf.pipeline", qos: .userInitiated)

    // throttle FPS
    private var lastStamp = CACurrentMediaTime()
    private var processing = false

    init() {
        do {
            detector = try Detector()
            status = "Modèle chargé (ANE)"
        } catch {
            status = "Erreur modèle : \(error)"
        }
    }

    /// Démarre l'analyse d'un fichier vidéo.
    func startFile(url: URL) {
        let src = FileFrameSource(url: url)
        attachAndStart(src, label: "Fichier : \(url.lastPathComponent)")
    }

    /// Démarre l'analyse du flux caméra.
    func startCamera() {
        let src = CameraFrameSource()
        attachAndStart(src, label: "Caméra live")
    }

    func stop() {
        source?.stop()
        source = nil
        status = "Arrêté"
    }

    private func attachAndStart(_ src: FrameSource, label: String) {
        stop()
        src.onFrame = { [weak self] frame in self?.handle(frame) }
        source = src
        status = label
        src.start()
    }

    private func handle(_ frame: Frame) {
        // Drop-frame : si l'ANE traite déjà, on saute (temps réel > exhaustivité).
        guard !processing, let detector else { return }
        processing = true

        work.async { [weak self] in
            guard let self else { return }
            let dets = (try? detector.detect(frame.pixelBuffer)) ?? []

            let now = CACurrentMediaTime()
            let dt = now - self.lastStamp
            self.lastStamp = now

            DispatchQueue.main.async {
                self.detections = dets
                self.count = dets.count
                if dt > 0 { self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) }
                self.processing = false
            }
        }
    }
}
