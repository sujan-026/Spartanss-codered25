import requests
import openrouteservice
import folium

# Function to calculate route using OpenRouteService
def get_route_with_avoidance(client, start_coords, end_coords, avoid_coords=None):
    try:
        params = {
            "coordinates": [start_coords, end_coords],
            "profile": "driving-car",
            "format": "geojson",
        }
        if avoid_coords:
            # Properly format avoid_polygons as GeoJSON
            params["options"] = {
                "avoid_polygons": {
                    "type": "MultiPolygon",
                    "coordinates": [[avoid_coords]],  # Note the extra nesting for MultiPolygon
                }
            }
        
        route = client.directions(**params)
        if "routes" in route and len(route["routes"]) > 0:
            return route
        else:
            raise Exception("No routes found")
    except Exception as e:
        raise Exception(f"OpenRouteService API error: {e}")

# Function to visualize the route using Folium
def visualize_route_with_avoidance(route, start_coords, end_coords, avoid_coords=None):
    # Create a map centered at the start location
    m = folium.Map(location=start_coords[::-1], zoom_start=10)

    # Add markers for start and end locations
    folium.Marker(location=start_coords[::-1], popup="Start").add_to(m)
    folium.Marker(location=end_coords[::-1], popup="End").add_to(m)

    # Add the route polyline
    coordinates = route["routes"][0]["geometry"]["coordinates"]
    folium.PolyLine(locations=[(lat, lon) for lon, lat in coordinates], color="blue", weight=5).add_to(m)

    # Add avoided area polygon if provided
    if avoid_coords:
        folium.Polygon(locations=[(lat, lon) for lon, lat in avoid_coords], color="red", fill=True, fill_opacity=0.4).add_to(m)

    # Save the map
    m.save("route_with_avoidance.html")
    print("Map saved as route_with_avoidance.html")

# Main function to calculate and visualize a route with avoidance
def main():
    # Predefined cities: Bengaluru and Chennai
    start_coords = [77.5946, 12.9716]  # Bengaluru
    end_coords = [80.2707, 13.0827]  # Chennai

    # Define an area to avoid (polygon around a hypothetical obstacle)
    avoid_coords = [
        [77.65, 12.95],
        [77.7, 12.95],
        [77.7, 12.98],
        [77.65, 12.98],
        [77.65, 12.95],
    ]

    ORS_API_KEY = "5b3ce3597851110001cf62488738ca77f9d74a4d9cfb84edda608120"  # Replace with your API key
    client = openrouteservice.Client(key=ORS_API_KEY)

    try:
        # Fetch route with avoidance
        print("Calculating route with avoidance...")
        route = get_route_with_avoidance(client, start_coords, end_coords, avoid_coords)
        print("Route successfully calculated.")

        # Visualize the route
        visualize_route_with_avoidance(route, start_coords, end_coords, avoid_coords)

    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
