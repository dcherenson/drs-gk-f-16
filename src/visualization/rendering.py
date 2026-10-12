import os

import moderngl as mgl
import numpy as np
import pygame as pg

from ..simulation import DemoCancelled
from .quaternion import Quaternion

dir_name = os.path.abspath(os.path.dirname(__file__))

def load_shader(ctx : mgl.Context, vertex_filename, frag_filename):
    with open(os.path.join(dir_name, vertex_filename)) as f:
        vertex_src = f.read()
    with open(os.path.join(dir_name, frag_filename)) as f:
        frag_src = f.read()

    return ctx.program(vertex_shader=vertex_src, fragment_shader=frag_src)

def load_mesh(ctx : mgl.Context, program, filename):
    v = []
    vn = []
    vertices = []
    indices = []

    with open(os.path.join(dir_name, filename), 'r') as file:
        for line in file:
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                vertex = [float(val) for val in values[1:4]]
                v.append(vertex)
            elif values[0] == 'vn':
                norm = [float(val) for val in values[1:4]]
                vn.append(norm)
            elif values[0] == 'vt':
                continue
            elif values[0] in ('usemtl', 'usemat'):
                continue
            elif values[0] == 'mtllib':
                continue
            elif values[0] == 'f':
                for val in values[1:]:
                    w = val.split('/')
                    vertex = np.hstack((v[int(w[0])-1], vn[int(w[2])-1]))
                    vertices.append(vertex)
                start = len(vertices) - len(values) + 1
                for i in range(start, len(vertices)-2):
                    indices.append([start, i+1, i+2])

    vbo = ctx.buffer(np.hstack(vertices).astype(np.float32).tobytes())
    ebo = ctx.buffer(np.hstack(indices).flatten().astype(np.uint32).tobytes())
    return ctx.simple_vertex_array(program, vbo, 'aPos', 'aNormal', index_buffer=ebo)


def load_heightfield_mesh(ctx: mgl.Context, program, x_coords, heights, z_coords, vertex_colors=None):
    """Create a shaded triangle mesh from a regular heightfield grid.

    Args:
        x_coords: 1D array of x positions (meters), length = cols.
        heights: 2D array of heights (meters), shape = (rows, cols).
        z_coords: 1D array of z positions (meters), length = rows.
    """
    x_coords = np.asarray(x_coords, dtype=np.float32)
    z_coords = np.asarray(z_coords, dtype=np.float32)
    heights = np.asarray(heights, dtype=np.float32)

    if heights.ndim != 2:
        raise ValueError(f"Expected 2D heights array, got shape {heights.shape}")

    rows, cols = heights.shape
    if x_coords.shape[0] != cols:
        raise ValueError("x_coords length must match heights columns")
    if z_coords.shape[0] != rows:
        raise ValueError("z_coords length must match heights rows")
    if rows < 2 or cols < 2:
        raise ValueError("Heightfield must have at least 2x2 samples")

    zz, xx = np.meshgrid(z_coords, x_coords, indexing='ij')
    yy = heights

    # Normal from y = f(x, z): n ~ (-dy/dx, 1, -dy/dz)
    dy_dz, dy_dx = np.gradient(yy, z_coords, x_coords, edge_order=1)
    normals = np.stack((-dy_dx, np.ones_like(yy), -dy_dz), axis=-1)
    normal_norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normal_norm[normal_norm < 1e-8] = 1.0
    normals = normals / normal_norm

    positions = np.stack((xx, yy, zz), axis=-1)

    if vertex_colors is not None:
        vertex_colors = np.asarray(vertex_colors, dtype=np.float32)
        if vertex_colors.shape != (rows, cols, 3):
            raise ValueError(
                f"vertex_colors must have shape {(rows, cols, 3)}, got {vertex_colors.shape}"
            )
        vertices = np.concatenate((positions, normals, vertex_colors), axis=-1).reshape(-1, 9).astype(np.float32)
    else:
        vertices = np.concatenate((positions, normals), axis=-1).reshape(-1, 6).astype(np.float32)

    indices = np.empty((rows - 1) * (cols - 1) * 6, dtype=np.uint32)
    k = 0
    for r in range(rows - 1):
        row0 = r * cols
        row1 = (r + 1) * cols
        for c in range(cols - 1):
            i0 = row0 + c
            i1 = i0 + 1
            i2 = row1 + c
            i3 = i2 + 1
            indices[k:k + 6] = (i0, i2, i1, i1, i2, i3)
            k += 6

    vbo = ctx.buffer(vertices.tobytes())
    ebo = ctx.buffer(indices.tobytes())
    if vertex_colors is not None:
        return ctx.simple_vertex_array(program, vbo, 'aPos', 'aNormal', 'aColor', index_buffer=ebo)
    return ctx.simple_vertex_array(program, vbo, 'aPos', 'aNormal', index_buffer=ebo)

def perspective(fov, aspect, near, far):
    fov *= np.pi/180
    right = -np.tan(fov/2) * near
    top = -right / aspect
    return np.array([[near/right,0,0,0],
                     [0,near/top,0,0],
                     [0,0,(far+near)/(far-near),-2*far*near/(far-near)],
                     [0,0,1,0]], dtype=np.float32)

class Transform:
    def __init__(self):
        self._position = np.zeros(3)
        self._rotation = Quaternion()
        self.scale = 1

    @property
    def position(self):
        return self._position.copy()

    @position.setter
    def position(self, position):
        self._position[:] = position

    @property
    def x(self):
        return self._position[0]

    @x.setter
    def x(self, x):
        self._position[0] = x

    @property
    def y(self):
        return self._position[1]

    @y.setter
    def y(self, y):
        self._position[1] = y

    @property
    def z(self):
        return self._position[2]

    @z.setter
    def z(self, z):
        self._position[2] = z

    @property
    def rotation(self):
        return self._rotation.copy()

    @rotation.setter
    def rotation(self, rotation):
        self._rotation._arr[:] = rotation._arr

    @property
    def matrix(self):
        matrix = np.eye(4)
        matrix[:3,:3] = self._rotation.mat().dot(np.eye(3)*self.scale)
        matrix[:3,3] = self._position
        return matrix

    @property
    def inv_matrix(self):
        matrix = np.eye(4)
        matrix[:3,3] = -self._position
        scale = np.eye(4)
        scale[:3,:3] /= self.scale
        matrix = scale.dot(matrix)
        rot = np.eye(4)
        rot[:3,:3] = self.rotation.inv().mat()
        matrix = rot.dot(matrix)
        return matrix


class RenderObject:
    def __init__(self, vao):
        self.vao = vao

        self.color = 1.0, 1.0, 1.0

        self.transform = Transform()

        self.draw_mode = mgl.TRIANGLES

    def render(self):
        self.vao.program['model'] = tuple(np.hstack(self.transform.matrix.T))
        try:
            self.vao.program['color'] = self.color
        except KeyError:
            pass
        self.vao.render(self.draw_mode)

class Grid(RenderObject):
    def __init__(self, ctx : mgl.Context, program, n, spacing):
        super().__init__(None)
        low = -(n-1)*spacing/2
        high = -low
        vertices = []
        indices = []
        for i in range(n):
            vertices.append([low + spacing*i, 0, low])
            vertices.append([low + spacing*i, 0,  high])
            indices.append([i*2, i*2+1])
        for i in range(n):
            vertices.append([low, 0, low + spacing*i])
            vertices.append([high, 0, low + spacing*i])
            indices.append([n*2+i*2, n*2+i*2+1])
        vertices = np.hstack(vertices)
        indices = np.hstack(indices)
        vbo = ctx.buffer(vertices.astype(np.float32).tobytes())
        ebo = ctx.buffer(indices.astype(np.uint32).tobytes())
        self.vao = ctx.simple_vertex_array(program, vbo, 'aPos', index_buffer=ebo)
        self.draw_mode = mgl.LINES

class Viewer:
    def __init__(self, width, height, headless=False):
        self.transform = Transform()
        self.width = width
        self.height = height

        self.headless = headless
        self._previous_sdl_video_driver = os.environ.get("SDL_VIDEODRIVER")
        if headless:
            # Prevent Pygame from trying to create a window/icon on macOS/Linux
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            pg.init()
            self.ctx = mgl.create_standalone_context()
            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.ctx.texture((width, height), 3)],
                depth_attachment=self.ctx.depth_renderbuffer((width, height))
            )
            self.fbo.use()
            self.display = None
        else:
            pg.init()
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
            pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 3)

            # Prefer OpenGL 3.3 core. Fall back to the platform default if the
            # explicit version request is rejected.
            try:
                pg.display.gl_set_attribute(pg.GL_CONTEXT_MAJOR_VERSION, 3)
                pg.display.gl_set_attribute(pg.GL_CONTEXT_MINOR_VERSION, 3)
                pg.display.gl_set_attribute(pg.GL_CONTEXT_PROFILE_MASK, pg.GL_CONTEXT_PROFILE_CORE)
                pg.display.gl_set_attribute(pg.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
                self.display = pg.display.set_mode((width, height), pg.DOUBLEBUF | pg.OPENGL)
            except pg.error:
                pg.display.quit()
                pg.display.init()
                pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 1)
                pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 3)
                self.display = pg.display.set_mode((width, height), pg.DOUBLEBUF | pg.OPENGL)

            self.ctx = mgl.create_context()
            self.fbo = self.ctx.screen

        self.ctx.enable(mgl.DEPTH_TEST)

        self.projection = perspective(90, width/height, .1, 1000.)

        self.prog = load_shader(self.ctx, "simple.vert", "simple.frag")
        self.prog['projection'] = tuple(np.hstack(self.projection.T))
        self.prog['lightDir'] = .6, -.8, 1.0

        self.unlit = load_shader(self.ctx, "simple.vert", "unlit.frag")
        self.unlit['projection'] = tuple(np.hstack(self.projection.T))

        self.terrain_prog = load_shader(self.ctx, "terrain.vert", "terrain.frag")
        self.terrain_prog['projection'] = tuple(np.hstack(self.projection.T))
        self.terrain_prog['lightDir'] = .6, -.8, 1.0

        self.set_view()

        self.objects = []
        self.set_caption("F-16 canyon demo")
        self._frame_texture = self._frame_program = self._frame_vao = self._frame_vbo = None
        if not headless:
            self._frame_texture = self.ctx.texture((width, height), 3)
            self._frame_program = self.ctx.program(
                vertex_shader="""#version 330
                    in vec2 position; in vec2 uv;
                    out vec2 texture_uv;
                    void main() { gl_Position = vec4(position, 0, 1); texture_uv = uv; }
                """,
                fragment_shader="""#version 330
                    uniform sampler2D frame;
                    in vec2 texture_uv; out vec4 color;
                    void main() { color = texture(frame, texture_uv); }
                """,
            )
            self._frame_program["frame"] = 0
            self._frame_vbo = self.ctx.buffer(np.asarray([
                -1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1,
            ], dtype=np.float32).tobytes())
            self._frame_vao = self.ctx.vertex_array(
                self._frame_program, [(self._frame_vbo, "2f 2f", "position", "uv")])



    def check_events(self):
        """Raise DemoCancelled for user cancellation events."""

        if self.headless:
            return
        for event in pg.event.get():
            if event.type == pg.QUIT:
                raise DemoCancelled("demo window closed")
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                raise DemoCancelled("demo cancelled with Escape")

    def set_caption(self, caption):
        if not self.headless and pg.get_init() and pg.display.get_init():
            pg.display.set_caption(str(caption))

    def set_view(self, x=None, y=None, z=None, rotation=None):
        if x is not None:
            self.transform.x = x
        if y is not None:
            self.transform.y = y
        if z is not None:
            self.transform.z = z
        if rotation is not None:
            self.transform.rotation = rotation
        self.prog['view'] = tuple(np.hstack(self.transform.inv_matrix.T))
        self.unlit['view'] = tuple(np.hstack(self.transform.inv_matrix.T))
        self.terrain_prog['view'] = tuple(np.hstack(self.transform.inv_matrix.T))

    def get_frame(self):
        data = self.fbo.read()
        return np.array(bytearray(data)).reshape(self.height, self.width,3)[-1::-1,:,:]

    def render(self, present=True):
        self.check_events()
        self.fbo.use()
        self.fbo.clear(0.5, 0.5, 0.5, 1.0)

        for obj in self.objects:
            obj.render()

        if present and not self.headless:
            pg.display.flip()

    def present_frame(self, frame):
        """Present an RGB frame on a human window after HUD composition."""

        if self.headless or self.display is None:
            return
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"frame must have shape {(self.height, self.width, 3)}")
        # Pygame surface blits do not update an OpenGL framebuffer. Upload the
        # composed frame as a texture so the live view includes HUD and paths.
        self.ctx.screen.use()
        self.ctx.disable(mgl.DEPTH_TEST)
        self._frame_texture.write(np.ascontiguousarray(frame[::-1]).tobytes())
        self._frame_texture.use(location=0)
        self._frame_vao.render(mgl.TRIANGLE_STRIP)
        self.ctx.enable(mgl.DEPTH_TEST)
        pg.display.flip()

    def close(self):
        # Release GL resources before the context disappears.  Some drivers
        # raise when an already released object is touched, hence the small
        # best-effort helper.
        def release(resource):
            method = getattr(resource, "release", None)
            if callable(method):
                try:
                    method()
                except Exception:
                    pass

        for obj in getattr(self, "objects", ()):
            release(getattr(obj, "vao", None))
        release(getattr(self, "prog", None))
        release(getattr(self, "unlit", None))
        release(getattr(self, "terrain_prog", None))
        for name in ("_frame_vao", "_frame_vbo", "_frame_program", "_frame_texture"):
            release(getattr(self, name, None))
        release(getattr(self, "fbo", None))
        release(getattr(self, "ctx", None))
        pg.quit()
        if self._previous_sdl_video_driver is None:
            os.environ.pop("SDL_VIDEODRIVER", None)
        else:
            os.environ["SDL_VIDEODRIVER"] = self._previous_sdl_video_driver

