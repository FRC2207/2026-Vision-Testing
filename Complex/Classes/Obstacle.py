class Obstacle:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def is_point_in_obstacle(self, pos_x: int, pos_y: int):
        # self.x and self.y is the top left corner
        return (self.x <= pos_x <= self.x + self.width) and (self.y <= pos_y <= self.y + self.height)

    def line_intersects_obstacle(self, line_start, line_end, rect_top_left, rect_dim):
        x1, y1 = line_start
        x2, y2 = line_end
        rx, ry = rect_top_left
        rw, rh = rect_dim
        
        # Rectangle boundaries
        left, top = rx, ry
        right, bottom = rx + rw, ry + rh

        # 1. Check if either end point is inside the rectangle
        def is_inside(px, py):
            return left <= px <= right and top <= py <= bottom

        if is_inside(x1, y1) or is_inside(x2, y2):
            return True

        # 2. Check intersection with the 4 sides of the rectangle
        # Sides: (x3, y3) to (x4, y4)
        sides = [
            ((left, top), (right, top)),    # Top
            ((right, top), (right, bottom)), # Right
            ((right, bottom), (left, bottom)), # Bottom
            ((left, bottom), (left, top))  # Left
        ]

        for p3, p4 in sides:
            if self.line_segments_intersect(line_start, line_end, p3, p4):
                return True

        return False

    def line_segments_intersect(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if denom == 0: return False # Parallel
        
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom

        # Intersection occurs if 0 <= ua <= 1 and 0 <= ub <= 1
        return 0 <= ua <= 1 and 0 <= ub <= 1