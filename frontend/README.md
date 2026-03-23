# Bedrock Model Profiler — Frontend

A React-based web application for exploring, analyzing, and comparing Amazon Bedrock foundation models with pricing, regional availability, and technical specifications.

**Live URL**: *(your CloudFront distribution URL after deployment)*

## Tech Stack

- **React 18** — UI framework
- **Vite** — Build tool with custom S3 proxy plugin for dev
- **Tailwind CSS v4** — Utility-first styling
- **Radix UI** — Accessible component primitives (Dialog, Popover, Select, Tabs, Tooltip, ScrollArea, etc.)
- **Zustand** — State management (3 stores: comparison, favorites, auth)
- **react-oidc-context** — AWS Cognito OIDC authentication
- **Lucide React** — Icon library
- **Recharts** — Analytics charts
- **Leaflet / react-leaflet** — Regional availability maps

## Features

- **Model Explorer**: Browse/filter/search 100+ models with 13 filter types, sortable, paginated, responsive grid
- **Model Detail**: Expanded modal with Technical Specs, Quotas, Pricing tabs; expandable availability pills showing In Region, CRIS, Batch, Mantle with inline detail sections
- **Model Comparison**: Up to 5 models side-by-side across 4 tabs (Overview, Pricing, Availability, Tech Specs)
- **Regional Availability**: Comprehensive model x region x consumption matrix
- **Favorites**: Persistent model shortlist (localStorage)
- **Dark/Light Theme**: Full theme support

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/              # Radix UI primitives (button, card, dialog, tooltip, etc.)
│   │   ├── layout/          # App shell (Layout, Sidebar, MainContent, ThemeProvider)
│   │   ├── models/          # Model Explorer, ModelCard, ModelCardExpanded, RegionalAvailability
│   │   └── comparison/      # ModelComparison + tabs (Overview, Pricing, Availability, TechSpecs)
│   ├── config/
│   │   ├── admin.js             # Permission functions (isAdmin, canViewQuotas, etc.)
│   │   ├── constants.js         # Consumption labels, provider colors
│   │   ├── generated-constants.js  # Auto-generated from backend config (sync-config.js)
│   │   └── dataSource.js        # Environment-aware data URLs
│   ├── hooks/
│   │   └── useModels.js         # Core data hook: fetches models + pricing, joins data
│   ├── stores/
│   │   ├── comparisonStore.js   # Selected models for comparison (persisted)
│   │   ├── favoritesStore.js    # Favorited model IDs (persisted)
│   │   └── authStore.js         # Cognito user state (session-only)
│   ├── utils/
│   │   ├── filters.js           # Filter logic: region, geo, status, capabilities, etc.
│   │   └── regionUtils.js       # Region metadata, geo sorting
│   ├── auth/
│   │   └── AuthGate.jsx         # OIDC authentication wrapper
│   └── lib/
│       └── utils.js             # cn() classname helper
├── scripts/
│   ├── deploy.sh                # S3 sync + CloudFront invalidation
│   └── sync-config.js           # Generate frontend constants from backend config
├── public/                      # Static assets (favicon)
└── dist/                        # Production build output
```

## Authentication & Authorization

The frontend uses AWS Cognito OIDC for authentication via react-oidc-context.

### Configuration

Configure via `.env` (copy from `template.env`):

```bash
VITE_COGNITO_AUTHORITY_URL=https://cognito-idp.us-east-1.amazonaws.com/us-east-1_xxxxx
VITE_COGNITO_CLIENT_ID=your-app-client-id
```

If environment variables are not set, the app runs without authentication.

### User Groups

User groups provide additive permissions:

- **beta-access-users**: Regional Availability, Roadmap (read), Quotas
- **region-roadmap-operators**: Roadmap editing
- **admins**: Analytics, Changelog, all features

### Permission Functions

Located in `config/admin.js`:

- `isAdmin(user)` — Check if user is admin
- `canViewRoadmap(user)` — Check if user can view roadmap
- `canViewRegionalAvailability(user)` — Check if user can view regional availability
- `canViewQuotas(user)` — Check if user can view quota data
- `canEditRoadmap(user)` — Check if user can edit roadmap
- `canViewAnalytics(user)` — Check if user can view analytics
- `canViewChangelog(user)` — Check if user can view changelog

Sidebar badges show user's highest group: `BETA` / `OP` / `ADM`

## Data Fetching

The `useModels.js` hook loads two JSON files in parallel:

- `/latest/bedrock_models.json` (~3MB)
- `/latest/bedrock_pricing.json` (~2MB)

### Environment-Aware Data Loading

- **Production**: Fetched from CloudFront `/latest/*` paths
- **Development**: Proxied via Vite S3 plugin (`/s3-data/*` → S3 bucket)

### Pricing Join Logic

The `getPricingForModel()` function matches model to pricing via `pricing_file_reference`. Handles 6 pricing types:

- `token`
- `image_generation`
- `video_generation`
- `video_second`
- `search_unit`
- `embedding`

## Development

```bash
npm install
npm run dev     # localhost:5173 with S3 proxy
npm run build   # Production build to dist/
npm run preview # Preview production build
```

The dev server proxies S3 data requests using your local AWS credentials configured via AWS CLI.

## Deployment

```bash
npm run build
./scripts/deploy.sh  # S3 sync + CloudFront invalidation
```

The deploy script syncs the `dist/` folder to S3 and invalidates the CloudFront cache.

## Key Components

| Component | File | Description |
|-----------|------|-------------|
| ModelExplorer | models/ModelExplorer.jsx | Main orchestrator: filters, sorting, pagination |
| ModelCard | models/ModelCard.jsx | Compact model card: pricing, context, modalities, tags |
| ModelCardExpanded | models/ModelCardExpanded.jsx | Detail modal: specs, quotas, pricing tabs + availability pills |
| RegionalAvailability | models/RegionalAvailability.jsx | Region x model availability matrix |
| ModelComparison | comparison/ModelComparison.jsx | Side-by-side comparison with 4 tab views |
| Layout | layout/Layout.jsx | App shell, sidebar, theme, confidential banner |

## Related

See `CLAUDE.md` in the repository root for complete project documentation including backend architecture, data pipeline, and deployment workflows.
