from maputils import *

class PozzalloPortToStationScene(TileMapScene):
    def create_tile_map(self, **kwargs):
        return TileMap(36.72522888144327, 14.84452309806404, 15, **kwargs)

    def construct(self):
        route = self.load_geojson("geojson/italy/pozzallo_port_to_station.geojson")
        self.play(MapMarker("Pozzallo Port", route.start).animate_creation())
        self.play(MapMarker("Pozzallo Station", route.end).animate_creation())
        route.create_and_animate(self)
