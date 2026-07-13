Front Next.js + shadcn/ui pour Boeuf Tracker.

## Demarrage

1. Lancer le backend Flask (racine du repo):

```bash
python app.py
```

2. Lancer le front Next.js:

```bash
cd frontend
pnpm dev
```

Le front tourne sur `http://localhost:3000` et proxifie automatiquement:
- `/api/*` -> `http://127.0.0.1:5000/api/*`
- `/video_feed` -> `http://127.0.0.1:5000/video_feed`

## Configuration backend optionnelle

Pour changer l'URL backend, definir:

```bash
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:5000
```

## Scripts

- `pnpm dev` : mode dev
- `pnpm build` : build production
- `pnpm lint` : lint frontend

## Notes

- Le backend Flask reste la source de verite pour la detection/segmentation/tracking.
- Cette UI remplace `templates/index.html` et `static/app.js` avec un stack Next.js + shadcn/ui.
