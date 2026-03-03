# AWS Region IATA Airport Codes Reference

This document maps AWS region codes to their corresponding IATA airport codes, used for display in the Bedrock Model Profiler Regional Availability page.

## Sources

### Primary Source (Authoritative)

**AWS Internal Region Code Reference**  
The authoritative source is an internal AWS spreadsheet that maps all AWS regions to their billing/IATA codes. This is the same source used across AWS services for billing region codes.

### Secondary Source (Partial - 23 regions only)

**AWS MediaConvert Billing Documentation**  
https://docs.aws.amazon.com/mediaconvert/latest/ug/usage-report-understand.html

This public documentation contains IATA codes for 23 regions where MediaConvert is available. However, it does NOT include newer regions like:
- eu-south-2 (Spain) - THC
- mx-central-1 (Mexico) - QRO
- eu-central-2 (Zurich) - ZRH
- ap-southeast-7 (Thailand) - BKK
- And others...

**⚠️ Do NOT use MediaConvert docs as the sole source - it's incomplete.**

## Complete IATA Code Mapping (34 Regions)

### Americas (NAMER)

| AWS Region | City | IATA | Airport/Location |
|------------|------|------|------------------|
| us-east-1 | N. Virginia | **IAD** | Washington Dulles International |
| us-east-2 | Ohio | **CMH** | John Glenn Columbus International |
| us-west-1 | N. California | **SFO** | San Francisco International |
| us-west-2 | Oregon | **PDX** | Portland International |
| ca-central-1 | Montreal | **YUL** | Montréal–Trudeau International |
| ca-west-1 | Calgary | **YYC** | Calgary International |

### Latin America (LATAM)

| AWS Region | City | IATA | Airport/Location |
|------------|------|------|------------------|
| sa-east-1 | São Paulo | **GRU** | São Paulo–Guarulhos International |
| mx-central-1 | Mexico | **QRO** | Querétaro Intercontinental |

### Europe (EMEA)

| AWS Region | City | IATA | Airport/Location |
|------------|------|------|------------------|
| eu-west-1 | Ireland | **DUB** | Dublin Airport |
| eu-west-2 | London | **LHR** | London Heathrow |
| eu-west-3 | Paris | **CDG** | Paris Charles de Gaulle |
| eu-central-1 | Frankfurt | **FRA** | Frankfurt Airport |
| eu-central-2 | Zurich | **ZRH** | Zurich Airport |
| eu-north-1 | Stockholm | **ARN** | Stockholm Arlanda |
| eu-south-1 | Milan | **MXP** | Milan Malpensa |
| eu-south-2 | Spain | **THC** | Tenerife South (AWS Spain DC code) |

### Middle East & Africa (EMEA)

| AWS Region | City | IATA | Airport/Location |
|------------|------|------|------------------|
| me-south-1 | Bahrain | **BAH** | Bahrain International |
| me-central-1 | UAE | **DXB** | Dubai International |
| il-central-1 | Tel Aviv | **TLV** | Ben Gurion Airport |
| af-south-1 | Cape Town | **CPT** | Cape Town International |

### Asia Pacific (APAC)

| AWS Region | City | IATA | Airport/Location |
|------------|------|------|------------------|
| ap-northeast-1 | Tokyo | **NRT** | Narita International |
| ap-northeast-2 | Seoul | **ICN** | Incheon International |
| ap-northeast-3 | Osaka | **KIX** | Kansai International |
| ap-southeast-1 | Singapore | **SIN** | Singapore Changi |
| ap-southeast-2 | Sydney | **SYD** | Sydney Kingsford Smith |
| ap-southeast-3 | Jakarta | **CGK** | Soekarno-Hatta International |
| ap-southeast-4 | Melbourne | **MEL** | Melbourne Airport |
| ap-southeast-5 | Malaysia | **KUL** | Kuala Lumpur International |
| ap-southeast-6 | Auckland | **AKL** | Auckland Airport |
| ap-southeast-7 | Thailand | **BKK** | Suvarnabhumi Airport |
| ap-south-1 | Mumbai | **BOM** | Chhatrapati Shivaji Maharaj International |
| ap-south-2 | Hyderabad | **HYD** | Rajiv Gandhi International |
| ap-east-1 | Hong Kong | **HKG** | Hong Kong International |
| ap-east-2 | Taipei | **TPE** | Taiwan Taoyuan International |

## GovCloud, China & Sovereign Regions (Not in Profiler)

| AWS Region | City | IATA | Notes |
|------------|------|------|-------|
| us-gov-west-1 | Oregon | **PDT** | AWS GovCloud (US-West) |
| us-gov-east-1 | Virginia | **OSU** | AWS GovCloud (US-East) |
| us-isob-east-1 | US ISO East | **DCA** | US ISO East |
| cn-north-1 | Beijing | **PEK** | AWS China (Beijing) |
| cn-northwest-1 | Ningxia | **ZHY** | AWS China (Ningxia) |
| eusc-de-east-1 | Germany | **THF** | European Sovereign Cloud |

## Notes on Specific Codes

### THC (eu-south-2 - Spain)
The code THC does not correspond to a major Spanish airport (Madrid = MAD, Barcelona = BCN). This appears to be an AWS-specific billing code, possibly derived from Tenerife South Airport or an internal designation.

### QRO (mx-central-1 - Mexico)
QRO is the IATA code for Querétaro Intercontinental Airport, not Mexico City (MEX). This indicates the AWS Mexico region data center is located near Querétaro, not Mexico City.

## Usage in Code

The IATA codes are stored in `backend/config/profiler-config.json` under `region_configuration.region_coordinates`:

```json
"us-east-1": {
  "lat": 38.9519,
  "lng": -77.448,
  "name": "N. Virginia",
  "geo": "US",
  "iata": "IAD"
}
```

Frontend access via `regionUtils.js`:

```javascript
import { getAirportCode } from '@/utils/regionUtils'

const iata = getAirportCode('us-east-1')  // Returns "IAD"
```

## Maintenance

When AWS adds new regions:
1. Check the AWS internal region code reference for the official IATA/billing code
2. If unavailable, check MediaConvert docs (may not have newer regions)
3. Add the code to `backend/config/profiler-config.json` in `region_coordinates`
4. Run `npm run sync-config` in the frontend directory
5. Update this documentation

## Last Updated

- **Date**: 2026-03-03
- **Updated by**: Claude (automated)
- **Changes**: Fixed eu-south-2 (MAD→THC) and mx-central-1 (MEX→QRO) based on AWS internal reference
