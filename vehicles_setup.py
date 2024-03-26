import random
import math
import numpy as np


class Vehicle():
    c=0
    def __init__(self):
        Vehicle.c+=1
        self.id=Vehicle.c
        
        self.computation_cycles_per_sec = 1  # 1 GHz
        self.transmission_power = random.randrange(1, 11) * 0.1  # Watt, 10 for RSU
        self.radius = random.randrange(5, 500)
        self.__latitude, self.__longitude, self.__altitude = self.generate_random_location()
        self.__pos = (self.__latitude, self.__longitude, self.__altitude)
        self.__absolute_speed, self.__cos_x, self.__cos_y, self.__cos_z = self.generate_random_speed_vector(40, 0, 0, 0)
        self.speed_vec = (self.__absolute_speed, self.__cos_x, self.__cos_y, self.__cos_z)
        self.task=(np.random.randint(500,  1001), 10,np.random.randint(2, 6))
        self.speed_vector = []

    def generate_random_location(self):
        # Generate random latitude (-90 to 90 degrees)
        latitude = random.uniform(-10, 10)
        # Generate random longitude (-180 to 180 degrees)
        longitude = random.uniform(-10, 10)
        # Generate random altitude (0 to 10000 meters)
        altitude = random.uniform(0, 10)  # Assuming the altitude range is from sea level to 10,000 meters
        return latitude, longitude, altitude

    def generate_random_speed_vector(self, initial_speed, initial_x, initial_y, initial_z):
        # Generate random absolute speed (0 to 100 meters per second)
        absolute_speed = random.uniform(initial_speed - 10, initial_speed + 10)
        if absolute_speed < 30:
            absolute_speed = 30
        if absolute_speed > 60:
            absolute_speed = 60
        # Generate random angles (0 to 2*pi radians)
        angle_x = random.uniform(initial_x - 5, initial_x + 5)
        if angle_x < 0:
            angle_x = 0
        if angle_x > 2 * math.pi:
            angle_x = 2 * math.pi
        angle_y = random.uniform(initial_y - 5, initial_y + 5)
        if angle_y < 0:
            angle_y = 0
        if angle_y > 2 * math.pi:
            angle_y = 2 * math.pi
        angle_z = random.uniform(initial_z - 5, initial_z + 5)
        if angle_z < 0:
            angle_z = 0
        if angle_z > 2 * math.pi:
            angle_z = 2 * math.pi
        # Calculate cosines of angles
        cos_x = math.cos(angle_x)
        cos_y = math.cos(angle_y)
        cos_z = math.cos(angle_z)
        return absolute_speed, cos_x, cos_y, cos_z

    def get_curr_pos(self, t=0):
        self.__absolute_speed, self.__cos_x, self.__cos_y, self.__cos_z = self.generate_random_speed_vector(
            self.__absolute_speed,
            self.__cos_x,
            self.__cos_y,
            self.__cos_z)
        # Calculate velocity components using absolute speed and direction cosines.
        vx = self.__absolute_speed * self.__cos_x
        vy = self.__absolute_speed * self.__cos_y
        vz = self.__absolute_speed * self.__cos_z
        self.__latitude = self.__latitude + (vx * t)
        self.__longitude = self.__longitude + (vy * t)
        self.__altitude = self.__altitude + (vz * t)
        return self.__latitude, self.__longitude, self.__altitude
    def get_speed_vector(self):
        vx = self.__absolute_speed * self.__cos_x
        vy = self.__absolute_speed * self.__cos_y
        vz = self.__absolute_speed * self.__cos_z
        return [vx,vy,vz]

    
    # def add_task(self, data_size, computation_capacity, max_delay):
    #     if self.id in tasks:
    #         tasks[self.id].append([data_size,computation_capacity, max_delay])
    #     else:
    #         tasks[self.id]=[[data_size, computation_capacity, max_delay]]


def create_vehicles(num_vehicles):
    
    vehicles = []
    for _ in range(num_vehicles):
        vehicle = Vehicle()
        vehicles.append(vehicle)
    return vehicles

# class Tasks():
#     global tasks
#     def __init__(self,tasks):
#         self.tasks =tasks
   
#     def add_task(self, user_id, data_size, computation_capacity, max_delay):
#         if user_id in tasks:
#             tasks[user_id].append([data_size,computation_capacity, max_delay])
#         else:
#             tasks[user_id]=[[data_size, computation_capacity, max_delay]]







def main():
    
    num_vehicles = int(input("Enter the number of vehicles: "))
    vehicles= create_vehicles(num_vehicles)
    print("Vehicles created successfully!")
    # for i, vehicle in enumerate(vehicles, start=1):
    #     print(f"Vehicle {i}: Computation Cycles/s: {vehicle.computation_cycles_per_sec}, Transmission Power: {vehicle.transmission_power}, Radius: {vehicle.radius}, Position: {vehicle.get_curr_pos()}, Speed_Vector: {vehicle.speed_vec}")
    print("vehciles is old script",vehicles)

    return vehicles
    
    # for vehicle in vehicles:
    #     vehicle.add_task(data_size=np.random.randint(500,  1001), cpu_cycles_per_bit=10, max_latency=np.random.randint(2, 6))

if __name__ == "__main__":
    c=1
    main()
