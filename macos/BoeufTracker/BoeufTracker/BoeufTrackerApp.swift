//
//  BoeufTrackerApp.swift
//  Phase 1 — Socle ANE : détection + segmentation + comptage des bœufs.
//
//  ANE-first : YOLO11-seg exporté en CoreML float16, exécuté via Vision
//  sur le Neural Engine. Aucune dépendance Python/webview.
//

import SwiftUI

@main
struct BoeufTrackerApp: App {
    var body: some Scene {
        WindowGroup {
            LiveView()
                .frame(minWidth: 900, minHeight: 600)
        }
        .windowStyle(.hiddenTitleBar)
    }
}
