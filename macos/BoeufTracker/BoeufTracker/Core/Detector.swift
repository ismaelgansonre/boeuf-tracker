//
//  Detector.swift
//  YOLO11-seg exécuté sur le Neural Engine via Vision.
//
//  ANE-first :
//   - Le modèle est un .mlpackage float16 (voir Conversion/export_yolo.py).
//   - MLModelConfiguration.computeUnits = .all → CoreML dispatche sur l'ANE.
//   - Vision (VNCoreMLRequest) gère le redimensionnement 640×640 et le
//     format pixel sans copie CPU superflue.
//
//  Phase 1 : on exploite les boîtes + le comptage. Les masques de
//  segmentation sont exposés pour la posture (Phase 3).
//

import Vision
import CoreML
import CoreVideo

/// Une détection brute d'un bœuf sur une frame.
struct Detection {
    let boundingBox: CGRect     // coordonnées normalisées Vision (origine bas-gauche)
    let confidence: Float
    let mask: CVPixelBuffer?    // masque d'instance (nil en Phase 1 si non exporté)
}

final class Detector {
    private let vnModel: VNCoreMLModel

    /// - Parameter modelName: nom du .mlpackage compilé (sans extension).
    init(modelName: String = "YOLO11Seg") throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all // laisse CoreML utiliser l'ANE

        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc")
                ?? Bundle.main.url(forResource: modelName, withExtension: "mlpackage") else {
            throw DetectorError.modelNotFound(modelName)
        }
        let mlModel = try MLModel(contentsOf: url, configuration: config)
        self.vnModel = try VNCoreMLModel(for: mlModel)
    }

    /// Analyse une frame et renvoie les bœufs détectés.
    /// Synchrone : à appeler depuis la file de la Pipeline, pas le main thread.
    func detect(_ pixelBuffer: CVPixelBuffer,
                confidenceThreshold: Float = 0.35) throws -> [Detection] {
        let request = VNCoreMLRequest(model: vnModel)
        request.imageCropAndScaleOption = .scaleFill // 640×640 fixe → ANE friendly

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        try handler.perform([request])

        // L'export Ultralytics CoreML expose des VNRecognizedObjectObservation
        // (NMS intégré si export avec nms=True).
        guard let results = request.results as? [VNRecognizedObjectObservation] else {
            return []
        }

        return results.compactMap { obs in
            guard let top = obs.labels.first,
                  top.identifier == "cow" || top.identifier == "boeuf" || top.identifier == "cattle",
                  top.confidence >= confidenceThreshold
            else { return nil }
            return Detection(boundingBox: obs.boundingBox,
                             confidence: top.confidence,
                             mask: nil) // masque branché en Phase 3
        }
    }

    enum DetectorError: Error, CustomStringConvertible {
        case modelNotFound(String)
        var description: String {
            switch self {
            case .modelNotFound(let n): return "Modèle CoreML introuvable dans le bundle : \(n)"
            }
        }
    }
}
