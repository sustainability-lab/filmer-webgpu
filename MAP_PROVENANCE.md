# Map boundary provenance

The application uses the official Survey of India **Outline of India**
international boundary vector, generalized at 1:16 million scale.

- Source page: <https://surveyofindia.gov.in/pages/outline-maps-of-india>
- Download: `Outline_of_India.zip`
- Download date: 24 July 2026
- ZIP SHA-256:
  `bf48477f01fe8addd6384490fc6f8decc9643110331ffef2c3f17e5cccd53b88`
- Source DBF date: 13 February 2026
- Source CRS: LCC/WGS84

Conversion:

```bash
ogr2ogr -f GeoJSON -t_srs EPSG:4326 -simplify 1500 \
  -lco RFC7946=YES -lco COORDINATE_PRECISION=5 \
  public/data/india-outline-soi.geojson Outline_of_India.shp
```

The RFC 7946 ring winding is reversed only at render time to match d3-geo’s
spherical polygon convention. Coordinates are not replaced with a third-party
international boundary dataset.

Survey of India states that its outline may be used for individual, internal,
educational, research, and website purposes and disallows commercial use. The
boundary file remains Survey of India copyright and is excluded from the MIT
code license. See the Survey of India source page and copyright notice for the
controlling terms.
