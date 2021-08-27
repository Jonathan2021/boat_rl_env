from Box2D import b2QueryCallback, b2ContactListener, b2RayCastCallback, b2Vec2


class ContactDetector(b2ContactListener):
    """ Collision handler callback. Handles all collisions from the Box2D world, sensors included. """
    def __init__(self):
        super().__init__()

    def BeginContact(self, contact):
        """ When a contact is initiated """
        # FIXME shorten the if / else logic. This is readable but not optimized
        if contact.fixtureA.sensor:
            if contact.fixtureB.sensor:
                contact.fixtureA.userData['touching_sensor'].append(contact.fixtureB.body) # Two sensors touching each other
            else:
                contact.fixtureA.userData['touching_hard'].append(contact.fixtureB.body) # Sensor touching a hard body

        # Same logic as above
        if contact.fixtureB.sensor:
            if contact.fixtureA.sensor:
                contact.fixtureB.userData['touching_sensor'].append(contact.fixtureA.body)
            else:
                contact.fixtureB.userData['touching_hard'].append(contact.fixtureA.body)

        if not contact.fixtureA.sensor and not contact.fixtureB.sensor: # 2 hard bodies touching
            contact.fixtureA.body.userData.hit_with.append(contact.fixtureB.body.userData)
            contact.fixtureB.body.userData.hit_with.append(contact.fixtureA.body.userData)

    def EndContact(self, contact):
        """ End of a contact """

        # Remove the contacts from the list (same logic as in Begin Contact but now we remove)
        if contact.fixtureA.sensor:
            if contact.fixtureB.sensor:
                contact.fixtureA.userData['touching_sensor'].remove(contact.fixtureB.body)
            else:
                contact.fixtureA.userData['touching_hard'].remove(contact.fixtureB.body)

        if contact.fixtureB.sensor:
            if contact.fixtureA.sensor:
                contact.fixtureB.userData['touching_sensor'].remove(contact.fixtureA.body)
            else:
                contact.fixtureB.userData['touching_hard'].remove(contact.fixtureA.body)

        if not contact.fixtureA.sensor and not contact.fixtureB.sensor:
            contact.fixtureA.body.userData.unhit(contact.fixtureB.body.userData)
            contact.fixtureB.body.userData.unhit(contact.fixtureA.body.userData)

class PlaceOccupied(b2QueryCallback):
    """ Query used with box2D AABB to know if fixtures are in the area """
    def __init__(self, ignore=[], ignore_type=[], dont_ignore=[]):
        b2QueryCallback.__init__(self)
        self.fixture = None
        self.ignore = ignore
        self.ignore_type = ignore_type
        self.dont_ignore = dont_ignore

    def ReportFixture(self, fixture):
        """ A fixture is found """
        data = fixture.body.userData
        if not data in self.dont_ignore and (data in self.ignore or data.type in self.ignore_type): # If we don't want to detect it
            return True # Continue the query
        self.fixture = fixture # Else register the fixture
        return False # Stop query

class LidarCallback(b2RayCastCallback):
    """
    This class captures the closest shape hit by a ray cast from p1 to p2).
    """
    def __init__(self, dont_report_type, dont_report_object, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None # Detected fixture
        self.fraction = 1 # Fraction of the original ray length
        self.dont_report_type = dont_report_type
        self.dont_report_object = dont_report_object

    def ReportFixture(self, fixture, point, normal, fraction):
        """ Called for each fixture found in the query. You control how the ray proceeds
        by returning a float that indicates the fractional length of the ray. By returning
        0, you set the ray length to zero. By returning the current fraction, you proceed
        to find the closest point. By returning 1, you continue with the original ray
         clipping.

         Here we return the fraction so we get the closest object (first to intercept ray).
         """

        if fixture.body.userData.type in self.dont_report_type or fixture.body.userData in self.dont_report_object or fixture.sensor: # Something we don't want to detec
            return -1 # Continue query
        self.fixture = fixture # Register fixture
        self.p2  = b2Vec2(point) # Register contact point
        self.normal = b2Vec2(normal)
        self.fraction = fraction # Register fraction of the original ray length
        return fraction

class CheckObstacleRayCallback(LidarCallback):
    """ Check if yes or no a ray cast hits an obstacle """
    def __init__(self, dont_report, dont_report_object=list(), **kwargs):
        super().__init__(dont_report, dont_report_object, **kwargs)
        self.hit_obstacle = False

    def ReportFixture(self, fixture, point, normal, fraction):
        res = super().ReportFixture(fixture, point, normal, fraction)
        self.hit_obstacle = not (res == -1 or res == 1) # True = it hit something

        if res == -1:
            return -1
        return res == 1 # Stop query if ray < 1 and different from -1 by setting ray length to 0.
