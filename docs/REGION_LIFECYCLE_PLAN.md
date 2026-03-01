# Region-Specific Lifecycle Status Implementation Plan

## Executive Summary

### What We're Building
A region-aware lifecycle tracking system that accurately reflects that models can have different lifecycle statuses (ACTIVE, LEGACY, EOL) in different AWS regions simultaneously.

### Why It Matters
The AWS Bedrock model lifecycle documentation shows that models transition through lifecycle stages at different times in different regions. For example:

**Claude 3.5 Sonnet v1** (as of today):
- **LEGACY** in: us-east-1, us-east-2, us-west-2, eu-central-1, eu-west-1, eu-west-3 (EOL: Mar 1, 2026)
- **ACTIVE** in: us-gov-east-1, us-gov-west-1, ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2

Currently, the system stores a **single global lifecycle status per model**, which:
1. Shows incorrect status for regions where the model is still ACTIVE
2. Loses EOL date information for specific regions
3. Provides incomplete migration guidance for users in different regions

---

## Current vs Target State

### Current Data Model

```json
{
  "model_lifecycle": {
    "status": "LEGACY",
    "eol_date": "March 1, 2026 (us-east-1, us-east-2, us-west-2 Regions)",
    "legacy_date": "September 4, 2025",
    "recommended_replacement": "Claude 3.5 Sonnet v2",
    "recommended_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
  }
}
```

**Problems:**
- Single status doesn't reflect regional differences
- EOL date string contains embedded region info that's not parseable
- No way to query "what is the status in ap-northeast-1?"

### Target Data Model

```json
{
  "model_lifecycle": {
    "global_status": "MIXED",
    "primary_status": "LEGACY",
    "regional_status": {
      "us-east-1": {
        "status": "LEGACY",
        "legacy_date": "September 4, 2025",
        "extended_access_date": "December 4, 2025",
        "eol_date": "March 1, 2026"
      },
      "us-east-2": {
        "status": "LEGACY",
        "legacy_date": "September 4, 2025",
        "extended_access_date": "December 4, 2025",
        "eol_date": "March 1, 2026"
      },
      "us-west-2": {
        "status": "LEGACY",
        "legacy_date": "September 4, 2025",
        "extended_access_date": "December 4, 2025",
        "eol_date": "March 1, 2026"
      },
      "ap-northeast-1": {
        "status": "ACTIVE",
        "launch_date": "August 7, 2024",
        "eol_date": "No sooner than July 30, 2025"
      },
      "ap-southeast-2": {
        "status": "ACTIVE",
        "launch_date": "August 5, 2024",
        "eol_date": "No sooner than July 30, 2025"
      }
    },
    "status_summary": {
      "ACTIVE": ["us-gov-east-1", "us-gov-west-1", "ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1", "ap-southeast-2"],
      "LEGACY": ["us-east-1", "us-east-2", "us-west-2", "eu-central-1", "eu-west-1", "eu-west-3"],
      "EOL": []
    },
    "recommended_replacement": "Claude 3.5 Sonnet v2",
    "recommended_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
  }
}
```

**Benefits:**
- Query status for any specific region
- Clear EOL dates per region
- Summary view for quick filtering
- Backward compatible (`primary_status` provides single value)

---

## Implementation Phases

### Phase 1: Lifecycle Collector Changes
**Goal:** Parse region-specific data from AWS documentation tables

### Phase 2: Data Model Schema Update
**Goal:** Define and validate new regional lifecycle schema

### Phase 3: Final Aggregator Changes
**Goal:** Merge regional lifecycle data into final model output

### Phase 4: Frontend Changes
**Goal:** Display region-specific lifecycle information in the UI

---

## Phase 1: Lifecycle Collector Changes

### File to Modify
`backend/lambdas/lifecycle-collector/handler.py`

### Current Behavior
The collector scrapes 3 tables (Active, Legacy, EOL) and stores:
- `regions`: Concatenated string (e.g., `"us-east-1us-east-2us-west-2"`)
- `launch_date`: First list item only
- `eol_date`: Full text including region parentheticals

### Required Changes

#### 1.1 Add Region Parsing Function

```python
import re
from typing import Optional

def parse_regions_from_cell(cell) -> list[str]:
    """Extract individual region names from a table cell.
    
    AWS docs format regions as:
    - Single region: "us-east-1"
    - Multiple as list items: "* us-east-1 * us-east-2 * us-west-2"
    - Multiple inline: "us-east-1, us-east-2, us-west-2"
    - With annotations: "* us-east-1\\* * us-east-2"
    """
    text = cell.get_text(strip=False)
    
    # AWS region pattern: xx-xxxx-N (with optional annotations like *)
    region_pattern = r'([a-z]{2}(?:-gov)?-[a-z]+-\d)'
    matches = re.findall(region_pattern, text.lower())
    
    # Deduplicate while preserving order
    seen = set()
    regions = []
    for region in matches:
        if region not in seen:
            seen.add(region)
            regions.append(region)
    
    return regions


def parse_dates_by_region(date_cell, regions: list[str]) -> dict[str, str]:
    """Map dates to their corresponding regions.
    
    AWS docs format dates as:
    - Single date: "3/14/2024"
    - Multiple as list items (parallel to regions):
      "* 3/14/2024 * 3/14/2024 * 8/30/2024"
    - Date with region qualifier:
      "March 1, 2026 (us-east-1, us-east-2, us-west-2 Regions)"
    """
    text = date_cell.get_text(strip=False)
    
    # Check for itemized list
    ul = date_cell.find("ul", class_="itemizedlist")
    if ul:
        items = ul.find_all("li")
        dates = [li.get_text(strip=True) for li in items]
        
        # Map dates to regions (parallel lists)
        date_by_region = {}
        for i, region in enumerate(regions):
            if i < len(dates):
                date_by_region[region] = dates[i]
            else:
                # Use last date for remaining regions
                date_by_region[region] = dates[-1] if dates else None
        return date_by_region
    
    # Check for parenthetical region qualifiers
    # Pattern: "Date (region1, region2, region3 Regions)"
    paren_pattern = r'([A-Za-z]+ \d+, \d{4})\s*\(([^)]+)\)'
    paren_matches = re.findall(paren_pattern, text)
    
    if paren_matches:
        date_by_region = {}
        for date_str, regions_str in paren_matches:
            qualified_regions = parse_regions_from_text(regions_str)
            for region in qualified_regions:
                if region in regions:
                    date_by_region[region] = date_str
        return date_by_region
    
    # Single date applies to all regions
    single_date = text.strip()
    if single_date:
        return {region: single_date for region in regions}
    
    return {}


def parse_regions_from_text(text: str) -> list[str]:
    """Extract regions from a text string (e.g., parenthetical qualifier)."""
    region_pattern = r'([a-z]{2}(?:-gov)?-[a-z]+-\d)'
    return re.findall(region_pattern, text.lower())
```

#### 1.2 Update parse_active_table()

```python
def parse_active_table(table) -> list[dict]:
    """Parse the Active models table with region-aware dates.
    
    Returns models with:
    - regions: list[str] - parsed region codes
    - regional_launch_dates: dict[region, date]
    - regional_eol_dates: dict[region, date]
    """
    models = []
    all_rows = table.find_all("tr")
    
    for row in all_rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        
        def get_cell_text(idx: int) -> str:
            if idx < len(cells):
                return cells[idx].get_text(strip=True)
            return ""
        
        # Parse regions as list
        regions = parse_regions_from_cell(cells[3]) if len(cells) > 3 else []
        
        # Parse dates mapped to regions
        regional_launch_dates = {}
        regional_eol_dates = {}
        
        if len(cells) > 4 and regions:
            regional_launch_dates = parse_dates_by_region(cells[4], regions)
        if len(cells) > 5 and regions:
            regional_eol_dates = parse_dates_by_region(cells[5], regions)
        
        model_data = {
            "provider": get_cell_text(0),
            "model_name": get_cell_text(1),
            "model_id": get_cell_text(2),
            "regions": regions,  # Now a list
            "regional_launch_dates": regional_launch_dates,
            "regional_eol_dates": regional_eol_dates,
            # Keep legacy fields for backward compatibility
            "launch_date": get_first_list_item_or_text(cells[4]) if len(cells) > 4 else None,
            "eol_date": get_cell_text(5) if len(cells) > 5 else None,
            "input_modalities": get_cell_text(6) if len(cells) > 6 else None,
            "output_modalities": get_cell_text(7) if len(cells) > 7 else None,
            "lifecycle_status": "active",
        }
        
        if model_data["model_id"]:
            models.append(model_data)
    
    return models
```

#### 1.3 Update parse_legacy_table()

```python
def parse_legacy_table(table) -> list[dict]:
    """Parse the Legacy models table with region-aware dates.
    
    Legacy table columns:
    - Model version
    - Legacy date (may have region qualifiers)
    - Public extended access date (may have region qualifiers)
    - EOL date (may have region qualifiers)
    - Recommended model version replacement
    - Recommended model ID
    """
    models = []
    all_rows = table.find_all("tr")
    
    for row in all_rows[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) < 4:
            continue
        
        def get_cell_text(idx: int) -> str:
            if idx < len(cells):
                return cells[idx].get_text(strip=True)
            return ""
        
        # Parse dates - legacy/EOL tables often have "(region1, region2 Regions)" format
        legacy_date_text = get_cell_text(1)
        extended_access_text = get_cell_text(2) if len(cells) > 5 else None
        eol_date_text = get_cell_text(3) if len(cells) > 5 else get_cell_text(2)
        
        # Extract regions from date fields
        regional_legacy_dates = parse_regional_dates_from_text(legacy_date_text)
        regional_extended_dates = parse_regional_dates_from_text(extended_access_text) if extended_access_text else {}
        regional_eol_dates = parse_regional_dates_from_text(eol_date_text)
        
        # Collect all mentioned regions
        all_regions = set(regional_legacy_dates.keys()) | set(regional_eol_dates.keys())
        if regional_extended_dates:
            all_regions |= set(regional_extended_dates.keys())
        
        model_data = {
            "model_name": get_cell_text(0),
            "legacy_date": legacy_date_text,
            "extended_access_date": extended_access_text,
            "eol_date": eol_date_text,
            "recommended_replacement": get_cell_text(4) if len(cells) > 5 else get_cell_text(3),
            "recommended_model_id": get_cell_text(5) if len(cells) > 5 else get_cell_text(4),
            "lifecycle_status": "legacy",
            # New regional fields
            "regions_affected": list(all_regions),
            "regional_legacy_dates": regional_legacy_dates,
            "regional_extended_dates": regional_extended_dates,
            "regional_eol_dates": regional_eol_dates,
            # Legacy fields
            "provider": None,
            "regions": None,
            "launch_date": None,
        }
        
        if model_data["model_name"]:
            models.append(model_data)
    
    return models


def parse_regional_dates_from_text(text: str) -> dict[str, str]:
    """Parse date text that may contain regional qualifiers.
    
    Handles formats:
    - "March 1, 2026 (us-east-1, us-east-2, us-west-2 Regions)"
    - "September 4, 2025" (no regions = applies globally)
    """
    if not text:
        return {}
    
    # Check for parenthetical region qualifier
    paren_pattern = r'([A-Za-z]+ \d+, \d{4})\s*\(([^)]+)\)'
    match = re.search(paren_pattern, text)
    
    if match:
        date_str = match.group(1)
        regions_str = match.group(2)
        regions = parse_regions_from_text(regions_str)
        return {region: date_str for region in regions}
    
    # No regional qualifier - return empty (caller handles global case)
    return {}
```

#### 1.4 Update Output Structure

```python
def scrape_lifecycle_data() -> dict:
    """Scrape and parse all lifecycle tables with regional awareness."""
    # ... existing scraping code ...
    
    # Build regional lookup structures
    regional_lifecycle_by_model = {}
    
    for model in all_models:
        model_id = model.get("model_id")
        model_name = model.get("model_name")
        status = model.get("lifecycle_status", "").upper()
        
        key = model_id if model_id else model_name
        if not key:
            continue
        
        if key not in regional_lifecycle_by_model:
            regional_lifecycle_by_model[key] = {
                "regional_status": {},
                "status_summary": {"ACTIVE": [], "LEGACY": [], "EOL": []},
                "recommended_replacement": model.get("recommended_replacement"),
                "recommended_model_id": model.get("recommended_model_id"),
            }
        
        entry = regional_lifecycle_by_model[key]
        
        # For Active table entries
        if status == "ACTIVE":
            regions = model.get("regions", [])
            for region in regions:
                entry["regional_status"][region] = {
                    "status": "ACTIVE",
                    "launch_date": model.get("regional_launch_dates", {}).get(region),
                    "eol_date": model.get("regional_eol_dates", {}).get(region),
                }
                if region not in entry["status_summary"]["ACTIVE"]:
                    entry["status_summary"]["ACTIVE"].append(region)
        
        # For Legacy/EOL table entries
        elif status in ["LEGACY", "EOL"]:
            regions = model.get("regions_affected", [])
            for region in regions:
                entry["regional_status"][region] = {
                    "status": status,
                    "legacy_date": model.get("regional_legacy_dates", {}).get(region),
                    "extended_access_date": model.get("regional_extended_dates", {}).get(region),
                    "eol_date": model.get("regional_eol_dates", {}).get(region),
                }
                if region not in entry["status_summary"][status]:
                    entry["status_summary"][status].append(region)
    
    return {
        "models": all_models,
        "models_by_id": models_by_id,
        "models_by_name": models_by_name,
        "regional_lifecycle_by_model": regional_lifecycle_by_model,
        "status_counts": status_counts,
        "total_models": len(all_models),
    }
```

### Acceptance Criteria - Phase 1
- [ ] Regions parsed as list instead of concatenated string
- [ ] Launch dates mapped to specific regions
- [ ] EOL dates mapped to specific regions
- [ ] Legacy/EOL regions extracted from parenthetical qualifiers
- [ ] New `regional_lifecycle_by_model` lookup structure created
- [ ] Backward compatible - existing fields still populated
- [ ] Unit tests for region parsing functions

---

## Phase 2: Data Model Schema Update

### File to Modify
`backend/lambdas/common/shared/validation.py` (if schema validation exists)

### Schema Definition

```python
# Regional lifecycle status schema
REGIONAL_LIFECYCLE_SCHEMA = {
    "type": "object",
    "properties": {
        "global_status": {
            "type": "string",
            "enum": ["ACTIVE", "LEGACY", "EOL", "MIXED"],
            "description": "Overall status (MIXED if varies by region)"
        },
        "primary_status": {
            "type": "string",
            "enum": ["ACTIVE", "LEGACY", "EOL"],
            "description": "Most common status (for backward compatibility)"
        },
        "regional_status": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["ACTIVE", "LEGACY", "EOL"]},
                    "launch_date": {"type": ["string", "null"]},
                    "legacy_date": {"type": ["string", "null"]},
                    "extended_access_date": {"type": ["string", "null"]},
                    "eol_date": {"type": ["string", "null"]}
                },
                "required": ["status"]
            }
        },
        "status_summary": {
            "type": "object",
            "properties": {
                "ACTIVE": {"type": "array", "items": {"type": "string"}},
                "LEGACY": {"type": "array", "items": {"type": "string"}},
                "EOL": {"type": "array", "items": {"type": "string"}}
            }
        },
        "recommended_replacement": {"type": ["string", "null"]},
        "recommended_model_id": {"type": ["string", "null"]},
        # Legacy fields for backward compatibility
        "status": {"type": "string"},
        "eol_date": {"type": ["string", "null"]},
        "legacy_date": {"type": ["string", "null"]},
        "release_date": {"type": ["string", "null"]}
    },
    "required": ["primary_status"]
}
```

### Acceptance Criteria - Phase 2
- [ ] Schema documented and validated
- [ ] Backward compatible with existing consumers
- [ ] TypeScript types generated for frontend (optional)

---

## Phase 3: Final Aggregator Changes

### File to Modify
`backend/lambdas/final-aggregator/handler.py`

### Current Behavior (lines 1452-1480)

```python
# Get model lifecycle (already in snake_case)
model_lifecycle = model.get("model_lifecycle", {})
if not model_lifecycle:
    model_lifecycle = {"status": "ACTIVE", "release_date": ""}

# Merge lifecycle data from scraper if available
if lifecycle_by_model:
    lifecycle_info = lifecycle_by_model.get(model_id, {})
    if lifecycle_info:
        scraped_status = lifecycle_info.get("lifecycle_status")
        if scraped_status:
            model_lifecycle["status"] = scraped_status.upper()
        # ... adds single eol_date, legacy_date, etc.
```

### Required Changes

#### 3.1 Update build_regional_lifecycle()

```python
def build_regional_lifecycle(
    model_id: str,
    model_name: str,
    regional_availability: list[str],
    regional_lifecycle_data: dict,
    lifecycle_by_model_id: dict,
    lifecycle_by_model_name: dict,
) -> dict:
    """Build comprehensive regional lifecycle data for a model.
    
    Merges data from:
    1. regional_lifecycle_by_model (parsed from docs with regional awareness)
    2. lifecycle_by_model_id (legacy lookup by model_id)
    3. lifecycle_by_model_name (legacy lookup by model_name)
    4. regional_availability (to ensure all available regions have status)
    
    Returns the new regional lifecycle schema.
    """
    # Try to get regional data (new format)
    regional_data = (
        regional_lifecycle_data.get(model_id) or
        regional_lifecycle_data.get(model_name) or
        {}
    )
    
    # Fall back to legacy lookups
    legacy_data = (
        lifecycle_by_model_id.get(model_id) or
        lifecycle_by_model_name.get(model_name) or
        {}
    )
    
    regional_status = {}
    status_counts = {"ACTIVE": 0, "LEGACY": 0, "EOL": 0}
    
    # First, populate from regional lifecycle data (new format)
    if regional_data.get("regional_status"):
        for region, region_info in regional_data["regional_status"].items():
            status = region_info.get("status", "ACTIVE").upper()
            regional_status[region] = {
                "status": status,
                "launch_date": region_info.get("launch_date"),
                "legacy_date": region_info.get("legacy_date"),
                "extended_access_date": region_info.get("extended_access_date"),
                "eol_date": region_info.get("eol_date"),
            }
            status_counts[status] = status_counts.get(status, 0) + 1
    
    # Fill in any missing regions from regional_availability as ACTIVE
    for region in regional_availability:
        if region not in regional_status:
            regional_status[region] = {
                "status": "ACTIVE",
                "launch_date": None,
                "legacy_date": None,
                "extended_access_date": None,
                "eol_date": None,
            }
            status_counts["ACTIVE"] += 1
    
    # If still empty but have legacy data, apply it globally
    if not regional_status and legacy_data:
        legacy_status = legacy_data.get("lifecycle_status", "active").upper()
        for region in regional_availability:
            regional_status[region] = {
                "status": legacy_status,
                "launch_date": legacy_data.get("launch_date"),
                "legacy_date": legacy_data.get("legacy_date"),
                "extended_access_date": legacy_data.get("extended_access_date"),
                "eol_date": legacy_data.get("eol_date"),
            }
            status_counts[legacy_status] = status_counts.get(legacy_status, 0) + 1
    
    # Determine global and primary status
    if status_counts["ACTIVE"] > 0 and (status_counts["LEGACY"] > 0 or status_counts["EOL"] > 0):
        global_status = "MIXED"
    elif status_counts["EOL"] > 0:
        global_status = "EOL"
    elif status_counts["LEGACY"] > 0:
        global_status = "LEGACY"
    else:
        global_status = "ACTIVE"
    
    # Primary status: most common, or highest severity if tied
    primary_status = max(
        ["ACTIVE", "LEGACY", "EOL"],
        key=lambda s: (status_counts.get(s, 0), ["ACTIVE", "LEGACY", "EOL"].index(s))
    )
    
    # Build status summary
    status_summary = {
        "ACTIVE": sorted([r for r, info in regional_status.items() if info["status"] == "ACTIVE"]),
        "LEGACY": sorted([r for r, info in regional_status.items() if info["status"] == "LEGACY"]),
        "EOL": sorted([r for r, info in regional_status.items() if info["status"] == "EOL"]),
    }
    
    return {
        "global_status": global_status,
        "primary_status": primary_status,
        "regional_status": regional_status,
        "status_summary": status_summary,
        "recommended_replacement": (
            regional_data.get("recommended_replacement") or
            legacy_data.get("recommended_replacement")
        ),
        "recommended_model_id": (
            regional_data.get("recommended_model_id") or
            legacy_data.get("recommended_model_id")
        ),
        # Legacy fields for backward compatibility
        "status": primary_status,
        "eol_date": legacy_data.get("eol_date"),
        "legacy_date": legacy_data.get("legacy_date"),
        "release_date": legacy_data.get("launch_date"),
    }
```

#### 3.2 Update transform_model_to_schema()

```python
def transform_model_to_schema(
    model_id: str,
    model: dict,
    regional_availability: list,
    # ... existing params ...
    lifecycle_by_model: dict = None,
    regional_lifecycle_data: dict = None,  # NEW
) -> dict:
    # ... existing code ...
    
    # Build regional lifecycle (NEW)
    model_lifecycle = build_regional_lifecycle(
        model_id=model_id,
        model_name=model.get("model_name", ""),
        regional_availability=regional_availability,
        regional_lifecycle_data=regional_lifecycle_data or {},
        lifecycle_by_model_id=lifecycle_by_model.get("models_by_id", {}) if lifecycle_by_model else {},
        lifecycle_by_model_name=lifecycle_by_model.get("models_by_name", {}) if lifecycle_by_model else {},
    )
    
    # ... rest of transform ...
    
    return {
        # ... existing fields ...
        "model_lifecycle": model_lifecycle,  # Now regional-aware
        # ...
    }
```

#### 3.3 Update lambda_handler()

```python
def lambda_handler(event: dict, context: Any) -> dict:
    # ... existing code ...
    
    # Read lifecycle data (updated to include regional data)
    lifecycle_s3_key = lifecycle_data_result.get("s3Key")
    lifecycle_data = (
        read_from_s3(s3_client, s3_bucket, lifecycle_s3_key)
        if lifecycle_s3_key
        else {"models_by_id": {}, "models_by_name": {}, "regional_lifecycle_by_model": {}}
    )
    
    lifecycle_by_model = lifecycle_data  # Full data structure
    regional_lifecycle_data = lifecycle_data.get("regional_lifecycle_by_model", {})
    
    # Pass to build_final_models
    final_providers = build_final_models(
        models_with_pricing,
        availability_data,
        token_specs_data,
        quotas_by_region,
        features_by_region,
        enriched_models_data,
        pricing_data,
        collection_timestamp,
        mantle_by_model,
        lifecycle_by_model,
        regional_lifecycle_data,  # NEW
    )
```

### Acceptance Criteria - Phase 3
- [ ] Regional lifecycle data merged into model output
- [ ] `global_status` correctly set to "MIXED" when statuses vary
- [ ] `primary_status` provides backward-compatible single status
- [ ] `status_summary` groups regions by status
- [ ] `regional_status` has entry for each available region
- [ ] Legacy fields (`status`, `eol_date`, etc.) still populated
- [ ] Models not in lifecycle tables default to ACTIVE

---

## Phase 4: Frontend Changes

### Files to Modify
1. `frontend/src/components/models/ModelCardExpanded.jsx` (LifecycleDetailsSection)
2. `frontend/src/components/models/ModelCard.jsx` (lifecycle badge, if exists)

### Current Behavior
Shows single lifecycle status badge and dates without regional context.

### Required Changes

#### 4.1 Update LifecycleDetailsSection Component

```jsx
// frontend/src/components/models/ModelCardExpanded.jsx

function LifecycleDetailsSection({ model, isLight }) {
  const lifecycle = model.model_lifecycle || {}
  
  // New regional-aware fields
  const globalStatus = lifecycle.global_status || lifecycle.status || 'ACTIVE'
  const primaryStatus = lifecycle.primary_status || lifecycle.status || 'ACTIVE'
  const regionalStatus = lifecycle.regional_status || {}
  const statusSummary = lifecycle.status_summary || {}
  
  // Legacy fields
  const releaseDate = lifecycle.release_date
  const recommendedReplacement = lifecycle.recommended_replacement
  
  // Check if this model has mixed regional status
  const isMixedStatus = globalStatus === 'MIXED'
  const hasRegionalData = Object.keys(regionalStatus).length > 0
  
  const [selectedRegion, setSelectedRegion] = useState(null)
  const [showAllRegions, setShowAllRegions] = useState(false)
  
  // Helper to format timestamp
  const formatTimestamp = (timestamp) => {
    if (!timestamp) return null
    if (typeof timestamp === 'number') {
      const date = new Date(timestamp * 1000)
      return date.toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
    }
    return timestamp
  }
  
  // Status colors
  const getStatusStyles = (status) => {
    const normalizedStatus = (status || 'ACTIVE').toUpperCase()
    switch (normalizedStatus) {
      case 'ACTIVE':
        return isLight
          ? 'bg-emerald-100 text-emerald-700 border border-emerald-200'
          : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
      case 'LEGACY':
        return isLight
          ? 'bg-amber-100 text-amber-700 border border-amber-200'
          : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
      case 'EOL':
        return isLight
          ? 'bg-red-100 text-red-700 border border-red-200'
          : 'bg-red-500/20 text-red-400 border border-red-500/30'
      case 'MIXED':
        return isLight
          ? 'bg-purple-100 text-purple-700 border border-purple-200'
          : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
      default:
        return isLight
          ? 'bg-stone-100 text-stone-700 border border-stone-200'
          : 'bg-white/10 text-slate-400 border border-white/20'
    }
  }
  
  const getStatusLabel = (status) => {
    const normalizedStatus = (status || 'ACTIVE').toUpperCase()
    switch (normalizedStatus) {
      case 'ACTIVE': return 'Active'
      case 'LEGACY': return 'Legacy'
      case 'EOL': return 'End of Life'
      case 'MIXED': return 'Mixed Status'
      default: return normalizedStatus
    }
  }
  
  return (
    <div className="space-y-3">
      {/* Global Status Badge */}
      <div className={cn(
        'rounded-lg p-3 border',
        isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
      )}>
        <div className="flex items-center justify-between">
          <p className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-300')}>
            {isMixedStatus ? 'Overall Status' : 'Status'}
          </p>
          <span className={cn(
            'px-2.5 py-1 rounded-full text-xs font-semibold',
            getStatusStyles(globalStatus)
          )}>
            {getStatusLabel(globalStatus)}
          </span>
        </div>
        
        {/* Mixed status explanation */}
        {isMixedStatus && (
          <p className={cn(
            'text-xs mt-2',
            isLight ? 'text-purple-600' : 'text-purple-400'
          )}>
            This model has different lifecycle statuses in different regions.
          </p>
        )}
      </div>
      
      {/* Regional Status Summary - only show if mixed or has regional data */}
      {(isMixedStatus || hasRegionalData) && (
        <div className={cn(
          'rounded-lg p-3 border',
          isLight ? 'bg-white border-stone-200' : 'bg-white/[0.02] border border-white/[0.06]'
        )}>
          <p className={cn('text-xs font-medium mb-2', isLight ? 'text-stone-600' : 'text-slate-300')}>
            Status by Region
          </p>
          
          <div className="space-y-2">
            {/* Active regions */}
            {statusSummary.ACTIVE?.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className={cn(
                    'inline-block w-2 h-2 rounded-full',
                    isLight ? 'bg-emerald-500' : 'bg-emerald-400'
                  )} />
                  <span className={cn('text-xs font-medium', isLight ? 'text-emerald-700' : 'text-emerald-400')}>
                    Active ({statusSummary.ACTIVE.length} regions)
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 ml-4">
                  {(showAllRegions ? statusSummary.ACTIVE : statusSummary.ACTIVE.slice(0, 5)).map(region => (
                    <Badge
                      key={region}
                      variant="outline"
                      className="text-[10px] cursor-pointer hover:bg-emerald-50"
                      onClick={() => setSelectedRegion(region)}
                    >
                      {regionDisplayNames[region] || region}
                    </Badge>
                  ))}
                  {!showAllRegions && statusSummary.ACTIVE.length > 5 && (
                    <button
                      onClick={() => setShowAllRegions(true)}
                      className={cn(
                        'text-[10px] px-1.5 py-0.5 rounded',
                        isLight ? 'text-emerald-600 hover:bg-emerald-50' : 'text-emerald-400 hover:bg-emerald-500/10'
                      )}
                    >
                      +{statusSummary.ACTIVE.length - 5} more
                    </button>
                  )}
                </div>
              </div>
            )}
            
            {/* Legacy regions */}
            {statusSummary.LEGACY?.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className={cn(
                    'inline-block w-2 h-2 rounded-full',
                    isLight ? 'bg-amber-500' : 'bg-amber-400'
                  )} />
                  <span className={cn('text-xs font-medium', isLight ? 'text-amber-700' : 'text-amber-400')}>
                    Legacy ({statusSummary.LEGACY.length} regions)
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 ml-4">
                  {statusSummary.LEGACY.map(region => (
                    <Badge
                      key={region}
                      variant="outline"
                      className="text-[10px] cursor-pointer hover:bg-amber-50"
                      onClick={() => setSelectedRegion(region)}
                    >
                      {regionDisplayNames[region] || region}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            
            {/* EOL regions */}
            {statusSummary.EOL?.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className={cn(
                    'inline-block w-2 h-2 rounded-full',
                    isLight ? 'bg-red-500' : 'bg-red-400'
                  )} />
                  <span className={cn('text-xs font-medium', isLight ? 'text-red-700' : 'text-red-400')}>
                    End of Life ({statusSummary.EOL.length} regions)
                  </span>
                </div>
                <div className="flex flex-wrap gap-1 ml-4">
                  {statusSummary.EOL.map(region => (
                    <Badge
                      key={region}
                      variant="outline"
                      className="text-[10px] cursor-pointer hover:bg-red-50"
                      onClick={() => setSelectedRegion(region)}
                    >
                      {regionDisplayNames[region] || region}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      
      {/* Selected Region Detail */}
      {selectedRegion && regionalStatus[selectedRegion] && (
        <div className={cn(
          'rounded-lg p-3 border',
          isLight ? 'bg-blue-50 border-blue-200' : 'bg-blue-500/10 border border-blue-500/20'
        )}>
          <div className="flex items-center justify-between mb-2">
            <span className={cn('text-xs font-medium', isLight ? 'text-blue-700' : 'text-blue-400')}>
              {regionDisplayNames[selectedRegion] || selectedRegion} ({selectedRegion})
            </span>
            <button
              onClick={() => setSelectedRegion(null)}
              className={cn('text-xs', isLight ? 'text-blue-600 hover:text-blue-700' : 'text-blue-400 hover:text-blue-300')}
            >
              Close
            </button>
          </div>
          
          <RegionLifecycleDetail
            regionInfo={regionalStatus[selectedRegion]}
            isLight={isLight}
          />
        </div>
      )}
      
      {/* Recommended Replacement */}
      {recommendedReplacement && (
        <div className={cn(
          'rounded-lg p-3 border',
          isLight ? 'bg-blue-50 border-blue-200' : 'bg-blue-500/10 border border-blue-500/20'
        )}>
          <p className={cn('text-xs', isLight ? 'text-blue-700' : 'text-blue-400')}>
            Recommended Replacement
          </p>
          <p className={cn('text-sm font-medium mt-1', isLight ? 'text-blue-800' : 'text-blue-300')}>
            {recommendedReplacement}
          </p>
        </div>
      )}
    </div>
  )
}

// Helper component for region detail
function RegionLifecycleDetail({ regionInfo, isLight }) {
  const { status, launch_date, legacy_date, extended_access_date, eol_date } = regionInfo
  
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2">
        <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Status:</span>
        <span className={cn('text-xs font-medium', getStatusColor(status, isLight))}>
          {status}
        </span>
      </div>
      
      {launch_date && (
        <div className="flex items-center gap-2">
          <span className={cn('text-xs', isLight ? 'text-stone-600' : 'text-slate-400')}>Launched:</span>
          <span className={cn('text-xs', isLight ? 'text-stone-800' : 'text-white')}>{launch_date}</span>
        </div>
      )}
      
      {legacy_date && (
        <div className="flex items-center gap-2">
          <span className={cn('text-xs', isLight ? 'text-amber-600' : 'text-amber-400')}>Legacy Date:</span>
          <span className={cn('text-xs', isLight ? 'text-amber-800' : 'text-amber-300')}>{legacy_date}</span>
        </div>
      )}
      
      {extended_access_date && (
        <div className="flex items-center gap-2">
          <span className={cn('text-xs', isLight ? 'text-orange-600' : 'text-orange-400')}>Extended Access:</span>
          <span className={cn('text-xs', isLight ? 'text-orange-800' : 'text-orange-300')}>{extended_access_date}</span>
        </div>
      )}
      
      {eol_date && (
        <div className="flex items-center gap-2">
          <span className={cn('text-xs', isLight ? 'text-red-600' : 'text-red-400')}>EOL Date:</span>
          <span className={cn('text-xs', isLight ? 'text-red-800' : 'text-red-300')}>{eol_date}</span>
        </div>
      )}
    </div>
  )
}

function getStatusColor(status, isLight) {
  switch (status?.toUpperCase()) {
    case 'ACTIVE': return isLight ? 'text-emerald-700' : 'text-emerald-400'
    case 'LEGACY': return isLight ? 'text-amber-700' : 'text-amber-400'
    case 'EOL': return isLight ? 'text-red-700' : 'text-red-400'
    default: return isLight ? 'text-stone-700' : 'text-slate-400'
  }
}
```

#### 4.2 Add Region Lifecycle Filter (Optional Enhancement)

```jsx
// Add to filter options in ModelFilters.jsx
{
  id: 'lifecycle_status',
  label: 'Lifecycle Status',
  options: [
    { value: 'active', label: 'Active (all regions)' },
    { value: 'mixed', label: 'Mixed Status' },
    { value: 'legacy', label: 'Legacy (any region)' },
    { value: 'eol', label: 'End of Life (any region)' },
  ]
}
```

### UI Mockups

#### Current View (Single Status)
```
┌─────────────────────────────────────────┐
│ Lifecycle Details                    ▼  │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │ Status              [  Legacy  ]    │ │
│ └─────────────────────────────────────┘ │
│ ┌─────────────────────────────────────┐ │
│ │ EOL Date                            │ │
│ │ March 1, 2026 (us-east-1, us-east-2,│ │
│ │ us-west-2 Regions)                  │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

#### Target View (Regional Status)
```
┌─────────────────────────────────────────┐
│ Lifecycle Details                    ▼  │
├─────────────────────────────────────────┤
│ ┌─────────────────────────────────────┐ │
│ │ Overall Status    [ Mixed Status ]  │ │
│ │ This model has different lifecycle  │ │
│ │ statuses in different regions.      │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ Status by Region                    │ │
│ │ ● Active (7 regions)                │ │
│ │   [us-gov-e1] [us-gov-w1] [ap-ne-1] │ │
│ │   [ap-ne-2] [ap-s-1] [ap-se-1]      │ │
│ │   [ap-se-2]                         │ │
│ │                                     │ │
│ │ ● Legacy (6 regions)                │ │
│ │   [us-east-1] [us-east-2] [us-west-2│ │
│ │   [eu-central-1] [eu-west-1]        │ │
│ │   [eu-west-3]                       │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ us-east-1 (N. Virginia)      [X]    │ │
│ │ Status: LEGACY                      │ │
│ │ Legacy Date: September 4, 2025      │ │
│ │ Extended Access: December 4, 2025   │ │
│ │ EOL Date: March 1, 2026             │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ Recommended Replacement             │ │
│ │ Claude 3.5 Sonnet v2                │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### Acceptance Criteria - Phase 4
- [ ] Global status badge shows "Mixed Status" for regional variation
- [ ] Status summary groups regions by lifecycle status
- [ ] Clicking a region shows detailed lifecycle info for that region
- [ ] Color coding consistent with status (green=Active, amber=Legacy, red=EOL)
- [ ] Backward compatible - works with old data format
- [ ] Mobile responsive layout

---

## Data Schema Examples

### Before (Current Format)

```json
{
  "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "model_lifecycle": {
    "status": "LEGACY",
    "eol_date": "March 1, 2026 (us-east-1, us-east-2, us-west-2 Regions)",
    "legacy_date": "September 4, 2025",
    "recommended_replacement": "Claude 3.5 Sonnet v2",
    "recommended_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"
  }
}
```

### After (Regional Format)

```json
{
  "model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "model_lifecycle": {
    "global_status": "MIXED",
    "primary_status": "LEGACY",
    "regional_status": {
      "us-east-1": {
        "status": "LEGACY",
        "launch_date": "June 20, 2024",
        "legacy_date": "September 4, 2025",
        "extended_access_date": "December 4, 2025",
        "eol_date": "March 1, 2026"
      },
      "us-east-2": {
        "status": "LEGACY",
        "launch_date": "June 20, 2024",
        "legacy_date": "September 4, 2025",
        "extended_access_date": "December 4, 2025",
        "eol_date": "March 1, 2026"
      },
      "us-west-2": {
        "status": "LEGACY",
        "launch_date": "June 20, 2024",
        "legacy_date": "September 4, 2025",
        "extended_access_date": "December 4, 2025",
        "eol_date": "March 1, 2026"
      },
      "ap-northeast-1": {
        "status": "ACTIVE",
        "launch_date": "August 7, 2024",
        "legacy_date": null,
        "extended_access_date": null,
        "eol_date": "No sooner than July 30, 2025"
      },
      "ap-southeast-2": {
        "status": "ACTIVE",
        "launch_date": "August 5, 2024",
        "legacy_date": null,
        "extended_access_date": null,
        "eol_date": "No sooner than July 30, 2025"
      },
      "us-gov-east-1": {
        "status": "ACTIVE",
        "launch_date": "July 30, 2024",
        "legacy_date": null,
        "extended_access_date": null,
        "eol_date": "No sooner than July 30, 2025"
      },
      "us-gov-west-1": {
        "status": "ACTIVE",
        "launch_date": "June 20, 2024",
        "legacy_date": null,
        "extended_access_date": null,
        "eol_date": "No sooner than July 30, 2025"
      }
    },
    "status_summary": {
      "ACTIVE": ["us-gov-east-1", "us-gov-west-1", "ap-northeast-1", "ap-northeast-2", "ap-south-1", "ap-southeast-1", "ap-southeast-2"],
      "LEGACY": ["us-east-1", "us-east-2", "us-west-2", "eu-central-1", "eu-west-1", "eu-west-3"],
      "EOL": []
    },
    "recommended_replacement": "Claude 3.5 Sonnet v2",
    "recommended_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "status": "LEGACY",
    "eol_date": "March 1, 2026 (us-east-1, us-east-2, us-west-2 Regions)",
    "legacy_date": "September 4, 2025",
    "release_date": "June 20, 2024"
  }
}
```

---

## Edge Cases

### 1. Model Only in Active Table
**Scenario:** Model appears only in Active table, not in Legacy or EOL tables.
**Solution:** All regions default to ACTIVE status with launch dates from Active table.

### 2. Model Only in Legacy/EOL Table
**Scenario:** Model appears in Legacy/EOL table but not in Active table (fully deprecated).
**Solution:** Extract regions from date parentheticals. If no regions specified, apply globally.

### 3. No Region Information
**Scenario:** Date fields don't contain region qualifiers.
**Solution:** Apply date to all regions where model is available (from regional-availability Lambda).

### 4. Region Not in Lifecycle Data
**Scenario:** Model available in region X but lifecycle table doesn't mention region X.
**Solution:** Default to ACTIVE status for that region (conservative assumption).

### 5. Conflicting Data Between Tables
**Scenario:** Same model/region appears in both Active and Legacy tables.
**Solution:** Legacy/EOL tables take precedence (they represent transitions).

### 6. Old Data Format in S3
**Scenario:** Frontend receives data in old format (no regional_status field).
**Solution:** Frontend falls back to single status display (backward compatible).

### 7. Mantle-Only Models
**Scenario:** Model exists only in Mantle API, not in Bedrock's ListFoundationModels.
**Solution:** Default to ACTIVE status with no lifecycle dates.

### 8. Future EOL Dates
**Scenario:** EOL date is "No sooner than X" (not a firm date).
**Solution:** Store as-is. UI indicates it's a minimum date, not confirmed.

---

## Estimated Effort

| Phase | Component | Tasks | Estimated Time |
|-------|-----------|-------|----------------|
| **1** | Lifecycle Collector | Region parsing functions, table updates, output structure | 6-8 hours |
| **2** | Data Model | Schema definition, validation, documentation | 2-3 hours |
| **3** | Final Aggregator | Merge logic, regional lifecycle builder, handler updates | 4-6 hours |
| **4** | Frontend | LifecycleDetailsSection component, region detail view | 6-8 hours |
| **Testing** | All | Unit tests, integration tests, manual verification | 4-6 hours |
| **Total** | | | **22-31 hours** |

### Recommended Sprint Plan
- **Sprint 1 (1 week):** Phases 1 & 2 - Backend parsing and schema
- **Sprint 2 (1 week):** Phase 3 - Aggregator changes and data output
- **Sprint 3 (1 week):** Phase 4 - Frontend display and testing

---

## Risks and Mitigations

### Risk 1: AWS Documentation Format Changes
**Risk:** AWS changes the HTML structure of lifecycle tables, breaking parsing.
**Probability:** Medium
**Impact:** High - data collection fails
**Mitigation:**
- Add defensive parsing with fallbacks
- Implement monitoring alerts for parsing failures
- Add test fixtures with sample HTML

### Risk 2: Data Volume Increase
**Risk:** Regional lifecycle data significantly increases JSON file size.
**Probability:** Low
**Impact:** Low - slight performance impact
**Mitigation:**
- ~100 models × ~20 regions × ~5 fields = ~10K additional entries
- Estimate: 50-100KB increase (acceptable)
- Add compression if needed

### Risk 3: Backward Compatibility Issues
**Risk:** Existing API consumers break with new schema.
**Probability:** Low
**Impact:** Medium
**Mitigation:**
- Keep legacy fields (`status`, `eol_date`, etc.)
- Add `primary_status` as single-value fallback
- Version the API if needed

### Risk 4: Incomplete Regional Data
**Risk:** Lifecycle tables don't have complete regional information.
**Probability:** Medium
**Impact:** Medium - some regions show default status
**Mitigation:**
- Default to ACTIVE for unlisted regions (safe assumption)
- Cross-reference with regional-availability data
- Log warnings for data gaps

### Risk 5: Performance Impact on Frontend
**Risk:** Rendering regional lifecycle data slows down model card.
**Probability:** Low
**Impact:** Low
**Mitigation:**
- Use collapsible sections (load on expand)
- Memoize status calculations
- Lazy load detailed region views

---

## Implementation Checklist

### Phase 1 Checklist
- [ ] Create `parse_regions_from_cell()` function
- [ ] Create `parse_dates_by_region()` function  
- [ ] Create `parse_regional_dates_from_text()` function
- [ ] Update `parse_active_table()` to return regional data
- [ ] Update `parse_legacy_table()` to extract regional dates
- [ ] Update `parse_eol_table()` to extract regional dates
- [ ] Update `scrape_lifecycle_data()` output structure
- [ ] Add unit tests for parsing functions
- [ ] Test with current AWS documentation HTML

### Phase 2 Checklist
- [ ] Document schema in code comments
- [ ] Add validation function (optional)
- [ ] Update DATA_SOURCES.md documentation

### Phase 3 Checklist
- [ ] Create `build_regional_lifecycle()` function
- [ ] Update `transform_model_to_schema()` parameters
- [ ] Update `build_final_models()` to pass regional data
- [ ] Update `lambda_handler()` to read new data structure
- [ ] Add unit tests for merge logic
- [ ] Deploy and verify S3 output

### Phase 4 Checklist
- [ ] Update `LifecycleDetailsSection` component
- [ ] Add `RegionLifecycleDetail` helper component
- [ ] Add region selection state management
- [ ] Add status summary display
- [ ] Add regional filter (optional)
- [ ] Test with both old and new data formats
- [ ] Verify mobile responsiveness
- [ ] Update any related components (ModelCard badge, etc.)

---

## Appendix: Reference Documentation

### AWS Model Lifecycle Documentation
- URL: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
- Contains: Active, Legacy, EOL tables with regional launch dates

### Related Files
- `backend/lambdas/lifecycle-collector/handler.py` - Source scraper
- `backend/lambdas/final-aggregator/handler.py` - Data merger
- `frontend/src/components/models/ModelCardExpanded.jsx` - UI display
- `backend/lambdas/regional-availability/handler.py` - Region data source

### Sample AWS Documentation HTML Structure
```html
<!-- Active table row example -->
<tr>
  <td>Anthropic</td>
  <td>Claude 3.5 Sonnet v1</td>
  <td>anthropic.claude-3-5-sonnet-20240620-v1:0</td>
  <td>
    <ul class="itemizedlist">
      <li>us-gov-east-1*</li>
      <li>us-gov-west-1</li>
      <li>ap-northeast-1</li>
    </ul>
  </td>
  <td>
    <ul class="itemizedlist">
      <li>6/20/2024</li>
      <li>7/30/2024</li>
      <li>8/7/2024</li>
    </ul>
  </td>
  <td>No sooner than 7/30/2025</td>
  <td>Text, Image</td>
  <td>Text, Chat</td>
</tr>

<!-- Legacy table row example -->
<tr>
  <td>Claude 3.5 Sonnet v1</td>
  <td>September 4, 2025</td>
  <td>December 4, 2025</td>
  <td>March 1, 2026 (us-east-1, us-east-2, us-west-2 Regions)</td>
  <td>Claude 3.5 Sonnet v2</td>
  <td>anthropic.claude-3-5-sonnet-20241022-v2:0</td>
</tr>
```
