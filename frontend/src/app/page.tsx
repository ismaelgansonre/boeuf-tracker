"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";

type CurrentSettings = {
  yolo_model?: string;
  imgsz?: number;
  embed_every?: number;
  threshold?: number;
  conf?: number;
};

type StatsResponse = {
  fps?: number;
  device?: string;
  frame_count?: number;
  source?: string;
  source_label?: string;
  active?: Array<{ name: string; conf: number }>;
  events?: string[];
  behavior?: Array<{ name: string; action: string; speed: number }>;
  current?: CurrentSettings;
  desired?: Partial<CurrentSettings>;
};

type SettingsResponse = {
  current: CurrentSettings;
  desired?: Partial<CurrentSettings>;
  models_available?: string[];
};

type DeviceResponse = {
  available: string[];
  current: string;
  gpus?: Array<{ index: number; id: string; name: string }>;
};

type VideoItem = {
  name: string;
  path: string;
  size_mb: number;
  source: "project" | "uploads";
};

const defaultCurrent: CurrentSettings = {
  yolo_model: "yolo11s-seg.pt",
  imgsz: 640,
  embed_every: 10,
  threshold: 0.65,
  conf: 0.4,
};

async function safeJson<T>(url: string, init?: RequestInit): Promise<T | null> {
  try {
    const res = await fetch(url, init);
    if (!res.ok) {
      return null;
    }
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export default function Home() {
  const streamRef = useRef<HTMLImageElement | null>(null);
  const streamTimerRef = useRef<number | null>(null);

  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [devices, setDevices] = useState<DeviceResponse | null>(null);
  const [settings, setSettings] = useState<SettingsResponse>({
    current: defaultCurrent,
    desired: {},
    models_available: [],
  });

  const [selectedVideoPath, setSelectedVideoPath] = useState("");
  const [selectedDevice, setSelectedDevice] = useState("auto");
  const [isResetOpen, setIsResetOpen] = useState(false);
  const [status, setStatus] = useState("");
  const [statusError, setStatusError] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const current = stats?.current ?? settings.current;
  const pending = useMemo(
    () => stats?.desired ?? settings.desired ?? {},
    [settings.desired, stats?.desired],
  );

  const refreshStream = useCallback(() => {
    if (!streamRef.current) return;
    streamRef.current.src = `/video_feed?t=${Date.now()}`;
  }, []);

  useEffect(() => {
    const streamImg = streamRef.current;
    if (!streamImg) return;

    const schedule = (delay: number) => {
      if (streamTimerRef.current) {
        window.clearTimeout(streamTimerRef.current);
      }
      streamTimerRef.current = window.setTimeout(refreshStream, delay);
    };

    streamImg.onload = () => schedule(40);
    streamImg.onerror = () => schedule(500);
    refreshStream();

    return () => {
      if (streamTimerRef.current) {
        window.clearTimeout(streamTimerRef.current);
      }
    };
  }, [refreshStream]);

  const loadVideos = useCallback(async () => {
    const data = await safeJson<{ videos: VideoItem[] }>("/api/videos");
    if (data?.videos) setVideos(data.videos);
  }, []);

  const loadDevices = useCallback(async () => {
    const data = await safeJson<DeviceResponse>("/api/devices");
    if (!data) return;
    setDevices(data);
    setSelectedDevice(data.current);
  }, []);

  const loadSettings = useCallback(async () => {
    const data = await safeJson<SettingsResponse>("/api/settings");
    if (!data) return;
    setSettings({
      current: { ...defaultCurrent, ...(data.current ?? {}) },
      desired: data.desired ?? {},
      models_available: data.models_available ?? [],
    });
  }, []);

  const refreshStats = useCallback(async () => {
    const data = await safeJson<StatsResponse>("/api/stats");
    if (data) {
      setStats(data);
    }
  }, []);

  useEffect(() => {
    const bootstrapId = window.setTimeout(() => {
      void loadVideos();
      void loadDevices();
      void loadSettings();
      void refreshStats();
    }, 0);
    const id = window.setInterval(refreshStats, 1000);
    return () => {
      window.clearTimeout(bootstrapId);
      window.clearInterval(id);
    };
  }, [loadDevices, loadSettings, loadVideos, refreshStats]);

  const setMessage = useCallback((message: string, isError = false) => {
    setStatus(message);
    setStatusError(isError);
    if (message) {
      window.setTimeout(() => setStatus(""), 3500);
    }
  }, []);

  const sliderToNumber = useCallback((value: number | readonly number[] | undefined, fallback: number) => {
    if (Array.isArray(value)) return value[0] ?? fallback;
    if (typeof value === "number") return value;
    return fallback;
  }, []);

  const pushSetting = useCallback(
    async (payload: Partial<CurrentSettings>) => {
      setIsLoading(true);
      const res = await safeJson<{ ok?: boolean; error?: string }>("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setIsLoading(false);
      if (!res?.ok) {
        setMessage(res?.error ?? "Mise a jour impossible", true);
        return;
      }
      await Promise.all([refreshStats(), loadSettings()]);
      setMessage("Parametres mis a jour");
    },
    [loadSettings, refreshStats, setMessage],
  );

  const pendingLabels = useMemo(() => {
    const labels: string[] = [];
    if (pending.imgsz != null) labels.push(`imgsz=${pending.imgsz}`);
    if (pending.embed_every != null) labels.push(`embed=${pending.embed_every}`);
    if (pending.threshold != null) labels.push(`th=${pending.threshold}`);
    if (pending.conf != null) labels.push(`conf=${pending.conf}`);
    if (pending.yolo_model != null) labels.push(`model=${pending.yolo_model}`);
    return labels;
  }, [pending]);

  const statsFps = Number(stats?.fps ?? 0);
  const statsFrameCount = stats?.frame_count ?? 0;
  const sourceLabel = stats?.source_label ?? stats?.source ?? "--";
  const deviceLabel = stats?.device ?? "--";
  const active = stats?.active ?? [];
  const events = stats?.events ?? [];
  const behavior = stats?.behavior ?? [];

  const modelValue = current.yolo_model ?? defaultCurrent.yolo_model!;
  const embedValue = current.embed_every ?? defaultCurrent.embed_every!;
  const thresholdValue = current.threshold ?? defaultCurrent.threshold!;
  const confValue = current.conf ?? defaultCurrent.conf!;
  const imgszValue = current.imgsz ?? defaultCurrent.imgsz!;

  return (
    <div className="min-h-screen bg-background text-foreground dark">
      <header className="border-b border-border bg-card/80 px-4 py-3">
        <div className="mx-auto flex w-full max-w-[1600px] items-center justify-between gap-4">
          <h1 className="text-lg font-bold tracking-wide">Reconnaissance Bovine en Temps Réel</h1>
          <div className="flex items-center gap-2 text-xs font-mono text-muted-foreground">
            <span>device: {deviceLabel}</span>
            <span>/</span>
            <span className={statsFps >= 20 ? "text-emerald-400" : statsFps >= 10 ? "text-amber-400" : "text-rose-400"}>
              fps: {statsFps.toFixed(1)}
            </span>
            <span>/</span>
            <span>frames: {statsFrameCount}</span>
            {pendingLabels.length > 0 ? (
              <>
                <span>/</span>
                <span className="text-amber-400">en attente: {pendingLabels.join(", ")}</span>
              </>
            ) : null}
          </div>
        </div>
      </header>

      <div className="border-b border-border bg-card px-4 py-3">
        <div className="mx-auto flex w-full max-w-[1600px] flex-wrap items-center gap-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Source</span>
          <select
            className="h-8 rounded-md border border-input bg-background px-2 text-xs font-mono"
            value={selectedVideoPath}
            onChange={async (event) => {
              const path = event.target.value;
              setSelectedVideoPath(path);
              if (!path) return;
              setMessage("Chargement video...");
              const res = await safeJson<{ ok?: boolean; error?: string }>("/api/source/file", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ path }),
              });
              if (!res?.ok) {
                setMessage(res?.error ?? "Bascule source impossible", true);
                return;
              }
              await refreshStats();
              setMessage("Source basculee");
            }}
          >
            <option value="">— Choisir une video du projet —</option>
            {videos.map((video) => (
              <option key={video.path} value={video.path}>
                [{video.source}] {video.name} ({video.size_mb} MB)
              </option>
            ))}
          </select>
          <label className="inline-flex cursor-pointer items-center">
            <input
              type="file"
              accept="video/*"
              className="hidden"
              onChange={async (event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                const formData = new FormData();
                formData.append("video", file);
                setMessage(`Upload ${file.name}...`);
                const res = await safeJson<{ ok?: boolean; error?: string }>("/api/upload-video", {
                  method: "POST",
                  body: formData,
                });
                if (!res?.ok) {
                  setMessage(res?.error ?? "Upload impossible", true);
                  return;
                }
                await loadVideos();
                setMessage("Video envoyee");
                event.target.value = "";
              }}
            />
            <Button variant="outline" size="sm">
              Upload
            </Button>
          </label>
          <Button
            variant="outline"
            size="sm"
            onClick={async () => {
              setMessage("Bascule webcam...");
              const res = await safeJson<{ ok?: boolean; error?: string }>("/api/source/webcam", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({}),
              });
              if (!res?.ok) {
                setMessage(res?.error ?? "Webcam indisponible", true);
                return;
              }
              await refreshStats();
              setMessage("Webcam active");
            }}
          >
            Webcam
          </Button>

          <Separator orientation="vertical" className="mx-1 h-6" />

          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Device</span>
          <select
            className="h-8 rounded-md border border-input bg-background px-2 text-xs font-mono"
            value={selectedDevice}
            onChange={async (event) => {
              const device = event.target.value;
              setSelectedDevice(device);
              setMessage("Bascule device...");
              const res = await safeJson<{ ok?: boolean; error?: string }>("/api/device", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ device }),
              });
              if (!res?.ok) {
                setMessage(res?.error ?? "Device indisponible", true);
                await loadDevices();
                return;
              }
              await Promise.all([refreshStats(), loadDevices()]);
              setMessage("Device mis a jour");
            }}
          >
            <option value="auto">Auto</option>
            {(devices?.available ?? []).map((device) => (
              <option key={device} value={device}>
                {device}
              </option>
            ))}
          </select>

          <div className="ml-auto flex items-center gap-2">
            {status ? (
              <span className={`text-xs font-mono ${statusError ? "text-rose-400" : "text-emerald-400"}`}>
                {status}
              </span>
            ) : null}
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                window.location.reload();
              }}
            >
              Recharger
            </Button>
            <Button
              variant="destructive"
              size="sm"
              onClick={async () => {
                setMessage("Redemarrage...");
                await safeJson("/api/restart", { method: "POST" });
                window.setTimeout(() => window.location.reload(), 3500);
              }}
            >
              Restart
            </Button>
            <Button variant="outline" size="sm" onClick={() => setIsResetOpen(true)}>
              Reset DB
            </Button>
          </div>
        </div>
      </div>

      <main className="mx-auto grid w-full max-w-[1600px] grid-cols-1 gap-4 p-4 lg:grid-cols-[1fr_360px]">
        <section className="flex min-h-0 flex-col">
          <div className="flex min-h-[420px] flex-1 items-center justify-center overflow-hidden rounded-xl border border-border bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img ref={streamRef} alt="Flux video" className="h-full w-full object-contain" />
          </div>
          <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-emerald-500" />
            bovin identifie
          </div>
        </section>

        <aside className="flex flex-col gap-3">
          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground">Parametres (live)</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="space-y-1">
                <label className="text-xs">Modele YOLO</label>
                <select
                  className="h-8 w-full rounded-md border border-input bg-background px-2 text-xs font-mono"
                  value={modelValue}
                  onChange={(event) => void pushSetting({ yolo_model: event.target.value })}
                  disabled={isLoading}
                >
                  {(settings.models_available ?? []).map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-xs">Resolution (imgsz)</label>
                <div className="grid grid-cols-5 gap-1">
                  {[320, 416, 640, 960, 1280].map((v) => (
                    <Button
                      key={v}
                      size="xs"
                      variant={imgszValue === v ? "default" : "outline"}
                      onClick={() => void pushSetting({ imgsz: v })}
                      disabled={isLoading}
                    >
                      {v}
                    </Button>
                  ))}
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-xs">Re-embed tous les N frames: {embedValue}</label>
                <Slider
                  min={1}
                  max={60}
                  step={1}
                  value={[embedValue]}
                  onValueChange={(value) => {
                    const v = sliderToNumber(value, embedValue);
                    setStats((prev) => (prev ? { ...prev, current: { ...prev.current, embed_every: v } } : prev));
                  }}
                  onValueCommitted={(value) => {
                    const v = sliderToNumber(value, embedValue);
                    if (typeof v === "number") {
                      void pushSetting({ embed_every: v });
                    }
                  }}
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs">Seuil Re-ID: {thresholdValue.toFixed(2)}</label>
                <Slider
                  min={0.3}
                  max={0.95}
                  step={0.01}
                  value={[thresholdValue]}
                  onValueChange={(value) => {
                    const v = sliderToNumber(value, thresholdValue);
                    setStats((prev) => (prev ? { ...prev, current: { ...prev.current, threshold: v } } : prev));
                  }}
                  onValueCommitted={(value) => {
                    const v = sliderToNumber(value, thresholdValue);
                    if (typeof v === "number") {
                      void pushSetting({ threshold: Number(v.toFixed(2)) });
                    }
                  }}
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs">Confiance YOLO: {confValue.toFixed(2)}</label>
                <Slider
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={[confValue]}
                  onValueChange={(value) => {
                    const v = sliderToNumber(value, confValue);
                    setStats((prev) => (prev ? { ...prev, current: { ...prev.current, conf: v } } : prev));
                  }}
                  onValueCommitted={(value) => {
                    const v = sliderToNumber(value, confValue);
                    if (typeof v === "number") {
                      void pushSetting({ conf: Number(v.toFixed(2)) });
                    }
                  }}
                />
              </div>

              <div className="flex justify-end">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={async () => {
                    setMessage("Re-match demande...");
                    const res = await safeJson<{ ok?: boolean; error?: string }>("/api/rematch", { method: "POST" });
                    if (!res?.ok) {
                      setMessage(res?.error ?? "Re-match impossible", true);
                      return;
                    }
                    setMessage("Re-match lance");
                  }}
                >
                  Re-match
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground">Animaux detectes</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-2 text-sm">
                <span className="font-bold text-emerald-400">{active.length}</span> actuellement visibles
              </p>
              <ul className="space-y-1.5">
                {active.length === 0 ? (
                  <li className="py-2 text-center text-sm italic text-muted-foreground">Aucun animal visible</li>
                ) : (
                  active.map((animal) => (
                    <li key={animal.track_id} className="flex items-center justify-between rounded-md bg-muted px-3 py-2 font-mono text-sm">
                      <span>{animal.name}</span>
                      <span className="text-muted-foreground">{(animal.conf * 100).toFixed(0)}%</span>
                    </li>
                  ))
                )}
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground">Journal</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1 font-mono text-xs">
                {events.length === 0 ? (
                  <li className="py-2 text-center italic text-muted-foreground">Aucun evenement</li>
                ) : (
                  events.map((eventLine, index) => (
                    <li key={`${eventLine}-${index}`} className="rounded border-l-2 border-border bg-muted px-3 py-1.5">
                      {eventLine}
                    </li>
                  ))
                )}
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-xs uppercase tracking-wider text-muted-foreground">Activites</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-1.5">
                {behavior.length === 0 ? (
                  <li className="py-2 text-center italic text-muted-foreground">Aucune activite</li>
                ) : (
                  behavior.map((item, index) => (
                    <li key={`${item.name}-${index}`} className="grid grid-cols-[1fr_auto_auto] items-center gap-2 rounded-md bg-muted px-3 py-2 font-mono text-xs">
                      <span>{item.name}</span>
                      <Badge variant="outline">{item.action}</Badge>
                      <span className="text-muted-foreground">{item.speed}px/s</span>
                    </li>
                  ))
                )}
              </ul>
            </CardContent>
          </Card>
        </aside>
      </main>

      <Dialog open={isResetOpen} onOpenChange={setIsResetOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reinitialiser la base ?</DialogTitle>
            <DialogDescription>
              Tous les bovins identifies seront oublies. Les prochaines detections repartiront a zero.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={() => setIsResetOpen(false)}>
              Annuler
            </Button>
            <Button
              variant="destructive"
              onClick={async () => {
                const res = await safeJson<{ ok?: boolean; error?: string }>("/api/db/reset", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({}),
                });
                if (!res?.ok) {
                  setMessage(res?.error ?? "Reset impossible", true);
                  return;
                }
                setIsResetOpen(false);
                await refreshStats();
                setMessage("Base purgee");
              }}
            >
              Purger
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
