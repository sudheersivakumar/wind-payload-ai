import math

def meters_to_km(value):
    return value / 1000

def km_to_meters(value):
    return value * 1000

def vector_magnitude(u, v):
    return math.sqrt(u**2 + v**2)
