//
//  LiveView.swift
//  Écran principal Phase 1 : choisir une source (fichier / caméra),
//  voir la vidéo, les boîtes des bœufs, le compteur et le FPS.
//

import SwiftUI
import UniformTypeIdentifiers

struct LiveView: View {
    @State private var pipeline = Pipeline()
    @State private var showImporter = false

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            detectionOverlay
        }
    }

    private var toolbar: some View {
        HStack(spacing: 12) {
            Button {
                showImporter = true
            } label: {
                Label("Ouvrir une vidéo", systemImage: "film")
            }

            Button {
                pipeline.startCamera()
            } label: {
                Label("Caméra live", systemImage: "video")
            }

            Button(role: .destructive) {
                pipeline.stop()
            } label: {
                Label("Stop", systemImage: "stop.fill")
            }

            Spacer()

            counter
        }
        .padding(12)
        .fileImporter(isPresented: $showImporter,
                      allowedContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie]) { result in
            if case .success(let url) = result {
                // accès sécurisé au fichier hors sandbox app
                _ = url.startAccessingSecurityScopedResource()
                pipeline.startFile(url: url)
            }
        }
    }

    private var counter: some View {
        HStack(spacing: 16) {
            Label("\(pipeline.count) bœufs", systemImage: "number.circle.fill")
                .font(.title3.monospacedDigit())
                .foregroundStyle(.tint)
            Text(String(format: "%.0f FPS", pipeline.fps))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Text(pipeline.status)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    /// Phase 1 : on dessine les boîtes normalisées par-dessus un fond.
    /// (Le rendu vidéo réel via AVSampleBufferDisplayLayer arrive à l'étape suivante ;
    ///  ici on visualise déjà les détections pour valider l'ANE.)
    private var detectionOverlay: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.opacity(0.9)
                ForEach(Array(pipeline.detections.enumerated()), id: \.offset) { _, det in
                    let r = det.boundingBox
                    // Vision : origine bas-gauche, normalisé → repère écran haut-gauche
                    let rect = CGRect(
                        x: r.minX * geo.size.width,
                        y: (1 - r.maxY) * geo.size.height,
                        width: r.width * geo.size.width,
                        height: r.height * geo.size.height
                    )
                    Rectangle()
                        .stroke(.green, lineWidth: 2)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                        .overlay(alignment: .topLeading) {
                            Text(String(format: "%.0f%%", det.confidence * 100))
                                .font(.caption2.bold())
                                .padding(2)
                                .background(.green)
                                .foregroundStyle(.black)
                                .position(x: rect.minX + 20, y: rect.minY + 8)
                        }
                }
            }
        }
    }
}
