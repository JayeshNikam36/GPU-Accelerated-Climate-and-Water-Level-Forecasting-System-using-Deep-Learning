# src/data_acquisition/iot_integration.py
import time
import random
from datetime import datetime
import logging
from typing import Dict, Generator

logger = logging.getLogger(__name__)

class IoTSimulator:
    """
    Simulates real-time sensor data stream (e.g., water level gage, temperature, precip).
    For development/testing until real IoT integration.
    Yields dicts every few seconds.
    """

    def __init__(self, site_name: str = "Passaic_River_PineBrook_Sim"):
        self.site_name = site_name
        self.base_level = 5.2  # feet (normal gage height)
        self.base_temp = 55.0  # °F
        self.last_time = datetime.now()

    def generate_reading(self) -> Dict:
        """Generate one simulated sensor reading."""
        now = datetime.now()
        delta_min = (now - self.last_time).total_seconds() / 60

        # Simulate slow changes + occasional spikes (rain/flood event)
        level_change = random.uniform(-0.05, 0.05) * delta_min
        if random.random() < 0.02:  # 2% chance of "storm spike"
            level_change += random.uniform(1.0, 3.0)

        temp_change = random.uniform(-0.5, 0.5)

        self.base_level = max(0.0, self.base_level + level_change)
        self.base_temp += temp_change

        reading = {
            "timestamp": now.isoformat(),
            "site": self.site_name,
            "gage_height_ft": round(self.base_level, 2),
            "temperature_f": round(self.base_temp, 1),
            "precip_in": round(random.uniform(0.0, 0.3) if random.random() < 0.1 else 0.0, 2),  # occasional rain
            "sensor_id": f"SIM-{random.randint(1000,9999)}",
            "quality": "P"  # Provisional, like USGS
        }

        self.last_time = now
        return reading

    def stream(self, interval_sec: float = 5.0, max_readings: int = None) -> Generator[Dict, None, None]:
        """
        Generator that yields readings every interval_sec seconds.
        Stop after max_readings or Ctrl+C.
        """
        count = 0
        while max_readings is None or count < max_readings:
            try:
                reading = self.generate_reading()
                yield reading
                logger.info(f"Simulated IoT reading: {reading}")
                time.sleep(interval_sec)
                count += 1
            except KeyboardInterrupt:
                logger.info("IoT simulation stopped by user.")
                break


# Quick test / demo when running file directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sim = IoTSimulator()
    print("Starting IoT simulation... Press Ctrl+C to stop.")
    for reading in sim.stream(interval_sec=3, max_readings=10):
        print(reading)