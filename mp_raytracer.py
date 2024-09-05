from PIL import Image
import sys
import math
import numpy as np
from numpy import linalg as LA

file = open(sys.argv[1], 'r')
content = file.read()

image = None
lines = content.split('\n')
output_file = ""
usingColor = True
hasTex = False
color = np.array([1.0,1.0,1.0])
texture = ""
texture_coordinate = np.array([0.0, 0.0])

eye = np.array([0.0, 0.0, 0.0])
forward = np.array([0.0, 0.0, -1.0])
right = np.array([1.0, 0.0, 0.0])
up = np.array([0.0, 1.0, 0.0])

width = 0
height = 0

# spheres = ['sphere', centerx, centery, centerz, rad]
# plane = ['plane', a,b,c,d]
objects = []
object_colors = []

textures = []

triangle_points = []

sun_dir = []
sun_color = []

width = 0
height = 0

expose = 0

fisheye = False
panorama = False



for line in lines:
    line = line.strip()
    # print(line)
    if line.startswith('png'):
        args = line.split()
        width = int(args[1])
        height = int(args[2])
        image = Image.new("RGBA", (int(args[1]), int(args[2])), (0,0,0,0))
        width = int(args[1])
        height = int(args[2])
        output_file = args[3]
    elif line.startswith('color'):
        args = line.split()
        color = np.array((float(args[1]), float(args[2]), float(args[3])))
    elif line.startswith('sphere'):
        args = line.split()
        objects.append(np.array([0, float(args[1]), float(args[2]), float(args[3]), float(args[4])]))
        if usingColor:
            object_colors.append(color)
            textures.append('none')
        else:
            textures.append(texture)
            object_colors.append(None)
    elif line.startswith('sun'):
        args = line.split()
        sun_dir.append(np.array([float(args[1]), float(args[2]), float(args[3])]))
        sun_color.append(color)
    elif line.startswith('expose'):
        args = line.split()
        expose = float(args[1])
    elif line.startswith('eye'):
        args = line.split()
        eye = np.array([float(args[1]), float(args[2]), float(args[3])])
    elif line.startswith('forward'):
        args = line.split()
        forward = np.array([float(args[1]), float(args[2]), float(args[3])])
        right = np.cross(forward, up)
        right /= LA.norm(right)
        up = np.cross(right, forward)
        up /= LA.norm(up)
    elif line.startswith('up'):
        args = line.split()
        up = np.array([float(args[1]), float(args[2]), float(args[3])])
        right = np.cross(forward, up)
        right /= LA.norm(right)
        up = np.cross(right, forward)
        up /= LA.norm(up)
    elif line.startswith('fisheye'):
        fisheye = True
    elif line.startswith('panorama'):
        panorama = True
    elif line.startswith('plane'):
        args = line.split()
        objects.append(np.array([1, float(args[1]), float(args[2]), float(args[3]), float(args[4])]))
        object_colors.append(color)
    elif line.startswith('xyz'):
        args = line.split()
        triangle_points.append(np.array([float(args[1]), float(args[2]), float(args[3]), texture_coordinate[0], texture_coordinate[1]]))
    elif line.startswith('texcoord'):
        args = line.split()
        texture_coordinate = np.array([float(args[1]), float(args[2])])
    elif line.startswith('tri'):
        args = line.split()
        to_add = []
        args = args[1:]
        for arg in args:
            arg = int(arg)
            if arg > 0:
                to_add.append(triangle_points[arg - 1])
            elif arg < 0:
                to_add.append(triangle_points[arg])
        objects.append(np.array([2, to_add], dtype=object))
        if usingColor:
            object_colors.append(color)
            textures.append('none')
        else:
            textures.append(texture)
            object_colors.append(None)
    elif line.startswith('texture'):
        args = line.split()
        hasTex = True
        if args[1] == 'none':
            usingColor = True
        else:
            usingColor = False
        texture = args[1]

def ray_sphere_intersect(ray_origin, ray_direction, sphere_center, sphere_radius):
    inside = LA.norm(sphere_center - ray_origin)**2 < (sphere_radius ** 2)
    tc = (np.dot(sphere_center - ray_origin, ray_direction))/LA.norm(ray_direction)
    if not inside and tc < 0:
        return None
    d2 = (LA.norm(ray_origin + (tc * ray_direction) - sphere_center))**2
    if not inside and sphere_radius ** 2 < d2:
        return None
    t_offset = np.sqrt(sphere_radius**2 - d2)/LA.norm(ray_direction)
    if inside:
        t = tc + t_offset
    else:
        t = tc - t_offset
    return t, t * ray_direction + ray_origin

def ray_plane_intersect(ray_origin, ray_direction, plane):
    
    normal = np.array(plane[0:3])
    normal /= LA.norm(normal)
    # if np.dot(ray_direction, normal) > 0:
    #     normal = normal * -1
    point_on_plane = (-plane[3] * (plane[0:3]))/((plane[0]**2) + (plane[1] ** 2) + (plane[2] ** 2))
    if np.dot(ray_direction, normal) == 0:
        return None
    t = (np.dot((point_on_plane - ray_origin), normal))/np.dot(ray_direction, normal)
    if t > 0:
        return t, t * ray_direction + ray_origin
    else:
        return None

def ray_triangle_intersect(ray_origin, ray_direction, vert1, vert2, vert3):
    edge1 = vert3[:3] - vert1[:3]
    edge2 = vert2[:3] - vert1[:3]
    normal = np.cross(edge1, edge2)
    normal /= LA.norm(normal)
    if np.dot(normal, ray_direction) == 0:
        return None

    t = (np.dot((vert1[:3] - ray_origin), normal))/(np.dot(ray_direction, normal))
    p = t * ray_direction + ray_origin 
    if t < 0:
        return None
    a1 = np.cross(edge1, normal)
    a2 = np.cross(edge2, normal)

    e1 = (1/(np.dot(a1, edge2))) * a1
    e2 = (1/(np.dot(a2, edge1))) * a2

    b1 = np.dot(e1,(p - vert1[:3]))
    b2 = np.dot(e2, (p - vert1[:3]))

    if b1 < 0 or b2 < 0 or b1 + b2 > 1:
        return None

    return t, p



def get_intersects(ray_origin, ray_direction):
    intersects = []
    for i, obj in enumerate(objects):
            if obj[0] == 0:
                curr = ray_sphere_intersect(ray_origin, ray_direction, np.array(obj[1:4]), obj[4])
                if curr:
                    intersects.append((i, curr[0], curr[1]))
            if obj[0] == 1:
                curr = ray_plane_intersect(ray_origin, ray_direction, np.array(obj[1:5]))
                if curr:
                    intersects.append((i, curr[0], curr[1]))
            if obj[0] == 2:
                curr = ray_triangle_intersect(ray_origin, ray_direction, obj[1][0], obj[1][1], obj[1][2])
                if curr:
                    intersects.append((i, curr[0], curr[1]))
    return intersects

def get_object_lighting(object_index, point):
    eye_to_point = (eye - point) * -1
    eye_to_point = eye_to_point / LA.norm(eye_to_point)
    if objects[object_index][0] == 0:
        surface_normal = (point - objects[object_index][1:4])
        surface_normal = surface_normal / LA.norm(surface_normal)
    elif objects[object_index][0] == 1:
        surface_normal = np.array(objects[object_index][1:4])
        surface_normal /= LA.norm(surface_normal)
    elif objects[object_index][0] == 2:
        triangle = objects[object_index][1]
        edge1 = triangle[2][:3] - triangle[0][:3]
        edge2 = triangle[1][:3] - triangle[0][:3]
        surface_normal = np.cross(edge1, edge2)
        surface_normal /= LA.norm(surface_normal)
    if np.dot(eye_to_point, surface_normal) > 0:
        surface_normal *= -1
    if not sun_dir:
        lambert_dot = np.dot([0,0,0], surface_normal)
        linear_color = [0,0,0] * objects[object_index][:3] * lambert_dot
        return linear_color
    linear_color = np.zeros(3)
    for i, sun in enumerate(sun_dir):
        potential_shadows = get_intersects(point, sun/LA.norm(sun))
        potential_shadows = [t for t in potential_shadows if t[0] != object_index]
        if not potential_shadows:
            dir_to_sun = sun / LA.norm(sun)
            lambert_dot = np.dot(dir_to_sun, surface_normal)
            if lambert_dot < 0:
                lambert_dot = 0
            if not hasTex or (textures and textures[object_index] == 'none'):
                linear_color += sun_color[i] * object_colors[object_index] * lambert_dot
            else:
                if objects[object_index][0] == 0:
                    norm = (1/objects[object_index][4]) * (point - objects[object_index][1:4])
                    longitude = math.atan2(norm[0], norm[2])
                    latitude = math.atan2(norm[1], math.sqrt(norm[0]**2 + norm[2]**2))
                    tex = Image.open(textures[object_index])
                    tex_width, tex_height = tex.size
                    x = int((tex_width/360.0) * (90 + math.degrees(longitude)))
                    y = int((tex_height/180.0) * (90 - math.degrees(latitude)))
                    rgba = tex.getpixel((x,y))
                    linear_rgb = []
                    for color in rgba[:3]:
                        color /= 255.0
                        if color <= 0.04045:
                            color /= 12.92
                        else:
                            color = ((color + 0.055)/1.055) ** 2.4
                        linear_rgb.append(color)
                    if len(linear_rgb) != 3:
                        print('ERROR')
                    linear_color += sun_color[i] * linear_rgb * lambert_dot
                elif objects[object_index][0] == 2:
                    triangle = objects[object_index][1]
                    vert1, vert2, vert3 = triangle
                    edge1 = vert3[:3] - vert1[:3]
                    edge2 = vert2[:3] - vert1[:3]
                    normal = np.cross(edge1, edge2)
                    normal /= LA.norm(normal)

                    a1 = np.cross(edge1, normal)
                    a2 = np.cross(edge2, normal)

                    e1 = (1/(np.dot(a1, edge2))) * a1
                    e2 = (1/(np.dot(a2, edge1))) * a2

                    b1 = np.dot(e1,(point - vert1[:3]))
                    b2 = np.dot(e2, (point - vert1[:3]))
                    b3 = 1 - b1 - b2
                    
                    texcord1 = vert1[3:5]
                    texcord2 = vert2[3:5]
                    texcord3 = vert3[3:5]

                    tex = Image.open(textures[object_index])
                    tex_width, tex_height = tex.size
                    x = (b1 * texcord2[0] + b2 * texcord3[0] + b3 * texcord1[0])
                    y = ((b1 * texcord2[1] + b2 * texcord3[1] + b3 * texcord1[1]))
                    x = math.floor((x) * tex_width)
                    y = math.floor((y) * tex_height )
                    if x >= tex_width:
                        x = tex_width - 1
                    if x < 0:
                        x = 0
                    if y >= tex_height:
                        y = tex_height -1
                    if y < 0:
                        y = 0
                    
                    
                    rgba = tex.getpixel((x,y))
                    linear_rgb = []
                    for color in rgba[:3]:
                        color /= 255.0
                        if color <= 0.04045:
                            color /= 12.92
                        else:
                            color = ((color + 0.055)/1.055) ** 2.4
                        linear_rgb.append(color)
                    if len(linear_rgb) != 3:
                        print('ERROR')
                    linear_color += sun_color[i] * linear_rgb * lambert_dot
    return linear_color



def linear_to_sRGB(linear_color):

    out = []
    for color in linear_color:
        if expose != 0:
            color = 1 - np.e ** (-expose * color)
        if color > 1:
            color = 1
        if color < 0:
            color = 0
        if color <= 0.0031308:
            color *= 12.92
        else:
            color = 1.055 * (color ** (1/2.4)) - 0.055
        out.append(math.floor(color * 255))
    return out


for x in range(width):
    for y in range(height):
        sx = (2 * x - width)/max(width, height)
        sy = (height - 2 * y)/max(width, height)
        if fisheye:
            if sx ** 2 + sy ** 2 <= 1:
                value = 1 - (sx ** 2) - (sy ** 2)
                if value >= 0:
                    fisheye_forward = math.sqrt(1-(sx**2) - (sy**2)) * forward
                    ray = fisheye_forward + sx * right + sy * up
                    ray /= LA.norm(ray)
                    t_vals = []
                    t_vals = get_intersects(eye, ray)
                    if t_vals:
                        min_t = min(t_vals, key=lambda x: x[1])
                        rgb = linear_to_sRGB(get_object_lighting(min_t[0],  min_t[2]))
                        image.im.putpixel((x,y), (rgb[0], rgb[1], rgb[2], 255))
        elif panorama:
            x_angle = math.radians(((360.0 / width) * x)-180)
            y_angle = math.radians(90-((180.0 / height) * y))
            ray_x = math.cos(y_angle) * math.sin(x_angle)
            ray_y = math.sin(y_angle)
            ray_z = math.cos(y_angle) * math.cos(x_angle)
            ray = ray_x * right + ray_y * up + ray_z * forward
            ray = ray / LA.norm(ray)
            t_vals = []
            t_vals = get_intersects(eye, ray)
            if t_vals:
                min_t = min(t_vals, key=lambda x: x[1])
                rgb = linear_to_sRGB(get_object_lighting(min_t[0], min_t[2]))
                image.im.putpixel((x,y), (rgb[0], rgb[1], rgb[2], 255))

        else:
            ray = forward + sx * right + sy * up
            ray = ray / LA.norm(ray)
            t_vals = []
            t_vals = get_intersects(eye, ray)
            if t_vals:
                min_t = min(t_vals, key=lambda x: x[1])
                rgb = linear_to_sRGB(get_object_lighting(min_t[0], min_t[2]))
                if x == 63 and y == 34:
                    rgb = linear_to_sRGB(get_object_lighting(min_t[0], min_t[2]))
                image.im.putpixel((x,y), (rgb[0], rgb[1], rgb[2], 255))
image.save(output_file)