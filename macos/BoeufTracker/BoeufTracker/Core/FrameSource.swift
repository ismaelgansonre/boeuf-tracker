//
//  FrameSource.swift
//  Source de frames unifiée : fichier vidéo OU caméra live.
//
//  Émet des CVPixelBuffer (format natif Vision/CoreML — zéro conversion
//  superflue avant l'ANE). Un seul protocole, deux implémentations,
//  la Pipeline ne sait pas d'où viennent les frames.
//

import AVFoundation
import CoreVideo

/// Une frame + son timestamp (secondes) pour le tracking temporel.
struct Frame {
    let pixelBuffer: CVPixelBuffer
    let time: Double
}

protocol FrameSource: AnyObject {
    /// Appelé pour chaque frame décodée/capturée, sur une file dédiée.
    var onFrame: ((Frame) -> Void)? { get set }
    func start()
    func stop()
}

// MARK: - Fichier vidéo

/// Lit un fichier vidéo frame par frame via AVAssetReader.
/// Décodage aussi vite que possible (analyse hors-ligne), pas de lecture "temps réel".
final class FileFrameSource: FrameSource {
    var onFrame: ((Frame) -> Void)?

    private let url: URL
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private let queue = DispatchQueue(label: "boeuf.filesource")
    private var running = false

    init(url: URL) { self.url = url }

    func start() {
        queue.async { [weak self] in self?.readLoop() }
    }

    func stop() {
        running = false
        reader?.cancelReading()
    }

    private func readLoop() {
        let asset = AVURLAsset(url: url)
        guard
            let track = try? loadFirstVideoTrack(asset),
            let reader = try? AVAssetReader(asset: asset)
        else { return }

        // BGRA : Vision accepte directement, pas de conversion couleur côté CPU.
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        out.alwaysCopiesSampleData = false
        guard reader.canAdd(out) else { return }
        reader.add(out)

        self.reader = reader
        self.output = out
        running = true
        reader.startReading()

        while running, reader.status == .reading,
              let sample = out.copyNextSampleBuffer() {
            if let pb = CMSampleBufferGetImageBuffer(sample) {
                let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sample))
                onFrame?(Frame(pixelBuffer: pb, time: t))
            }
        }
        running = false
    }

    private func loadFirstVideoTrack(_ asset: AVURLAsset) throws -> AVAssetTrack? {
        // API synchrone volontaire ici (thread dédié). En prod : loadTracks(withMediaType:).
        asset.tracks(withMediaType: .video).first
    }
}

// MARK: - Caméra live

/// Capture caméra (webcam / caméra USB de ferme) via AVCaptureSession.
final class CameraFrameSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {
    var onFrame: ((Frame) -> Void)?

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "boeuf.camerasource")

    override init() {
        super.init()
        configure()
    }

    private func configure() {
        session.beginConfiguration()
        session.sessionPreset = .high

        if let device = AVCaptureDevice.default(for: .video),
           let input = try? AVCaptureDeviceInput(device: device),
           session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true // on ne bufferise pas : temps réel
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) { session.addOutput(output) }

        session.commitConfiguration()
    }

    func start() { queue.async { [weak self] in self?.session.startRunning() } }
    func stop()  { session.stopRunning() }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let t = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        onFrame?(Frame(pixelBuffer: pb, time: t))
    }
}
