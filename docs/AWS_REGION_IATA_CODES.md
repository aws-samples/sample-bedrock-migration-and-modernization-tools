# AWS Region IATA Airport Codes Reference

This document maps AWS region codes to their corresponding IATA airport codes, used for display in the Bedrock Model Profiler Regional Availability page.

## Primary Source

**AWS MediaConvert Billing Documentation**  
https://docs.aws.amazon.com/mediaconvert/latest/ug/usage-report-understand.html

This AWS documentation provides the official mapping between AWS billing region codes (IATA codes) and AWS region codes for MediaConvert usage reports.

## Secondary Source (for regions not in MediaConvert docs)

For regions not listed in the MediaConvert documentation, standard IATA airport codes were used based on the primary international airport serving each AWS region's city.

**IATA Official Code Search**  
https://www.iata.org/en/publications/directories/code-search/

## Complete IATA Code Mapping

### From AWS MediaConvert Documentation (Official)

| AWS Region | City | IATA | Airport Name |
|------------|------|------|--------------|
| us-east-1 | N. Virginia | IAD | Washington Dulles International |
| us-east-2 | Ohio | CMH | John Glenn Columbus International |
| us-west-1 | N. California | SFO | San Francisco International |
| us-west-2 | Oregon | PDX | Portland International |
| eu-west-1 | Ireland | DUB | Dublin Airport |
| eu-west-2 | London | LHR | London Heathrow |
| eu-west-3 | Paris | CDG | Paris Charles de Gaulle |
| eu-central-1 | Frankfurt | FRA | Frankfurt Airport |
| eu-north-1 | Stockholm | ARN | Stockholm Arlanda |
| ap-south-1 | Mumbai | BOM | Chhatrapati Shivaji Maharaj International |
| ap-northeast-1 | Tokyo | NRT | Narita International |
| ap-northeast-2 | Seoul | ICN | Incheon International |
| ap-northeast-3 | Osaka | KIX | Kansai International |
| ap-southeast-1 | Singapore | SIN | Singapore Changi |
| ap-southeast-2 | Sydney | SYD | Sydney Kingsford Smith |
| ap-southeast-4 | Melbourne | MEL | Melbourne Airport |
| sa-east-1 | São Paulo | GRU | São Paulo–Guarulhos International |
| ca-central-1 | Montreal | YUL | Montréal–Trudeau International |
| me-central-1 | UAE | DXB | Dubai International |
| af-south-1 | Cape Town | CPT | Cape Town International |

### Standard IATA Codes (Not in MediaConvert docs)

| AWS Region | City | IATA | Airport Name | Source |
|------------|------|------|--------------|--------|
| eu-central-2 | Zurich | ZRH | Zurich Airport | Standard IATA |
| eu-south-1 | Milan | MXP | Milan Malpensa | Standard IATA |
| eu-south-2 | Spain | MAD | Madrid Barajas | Standard IATA |
| ap-south-2 | Hyderabad | HYD | Rajiv Gandhi International | Standard IATA |
| ap-southeast-3 | Jakarta | CGK | Soekarno-Hatta International | Standard IATA |
| ap-southeast-5 | Malaysia | KUL | Kuala Lumpur International | Standard IATA |
| ap-southeast-6 | Auckland | AKL | Auckland Airport | Standard IATA |
| ap-southeast-7 | Bangkok | BKK | Suvarnabhumi Airport | Standard IATA |
| ap-east-1 | Hong Kong | HKG | Hong Kong International | Standard IATA |
| ap-east-2 | Taipei | TPE | Taiwan Taoyuan International | Standard IATA |
| ca-west-1 | Calgary | YYC | Calgary International | Standard IATA |
| me-south-1 | Bahrain | BAH | Bahrain International | Standard IATA |
| il-central-1 | Tel Aviv | TLV | Ben Gurion Airport | Standard IATA |
| mx-central-1 | Mexico City | MEX | Mexico City International | Standard IATA |

## GovCloud and China Regions (Not Included)

The following regions are not included in the Bedrock Model Profiler as they are isolated partitions:

| AWS Region | City | IATA | Notes |
|------------|------|------|-------|
| us-gov-west-1 | Oregon | PDT | AWS GovCloud (US-West) |
| us-gov-east-1 | Virginia | - | AWS GovCloud (US-East) |
| cn-north-1 | Beijing | PEK | AWS China (Beijing) |
| cn-northwest-1 | Ningxia | ZHY | AWS China (Ningxia) |

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

## Last Updated

- **Date**: 2026-03-03
- **Updated by**: Claude (automated)
- **Reason**: Initial creation for Regional Availability tooltip enhancement
