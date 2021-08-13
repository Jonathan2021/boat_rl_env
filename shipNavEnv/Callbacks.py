from Box2D import b2QueryCallback, b2ContactListener, b2RayCastCallback, b2Vec2


# collision handler
class ContactDetector(b2ContactListener):
    def __init__(self):
        super().__init__()

    def BeginContact(self, contact):
        #print("In begin Contact")
        #print("Is contact fixture A a sensor %s" % contact.fixtureA.sensor)
        #print("Is contact fixture B a sensor %s" % contact.fixtureB.sensor)
        if contact.fixtureA.sensor:
            contact.fixtureA.userData['touching'].append(contact.fixtureB.body)
        if contact.fixtureB.sensor:
            contact.fixtureB.userData['touching'].append(contact.fixtureA.body)
        if not contact.fixtureA.sensor and not contact.fixtureB.sensor:
            contact.fixtureA.body.userData.hit_with.append(contact.fixtureB.body.userData)
            contact.fixtureB.body.userData.hit_with.append(contact.fixtureA.body.userData)
        #print('There was a contact!')

    def EndContact(self, contact):
        #print("Ending Contact")
        if contact.fixtureA.sensor:
            contact.fixtureA.userData['touching'].remove(contact.fixtureB.body)
        if contact.fixtureB.sensor:
            contact.fixtureB.userData['touching'].remove(contact.fixtureA.body)
        if not contact.fixtureA.sensor and not contact.fixtureB.sensor:
            contact.fixtureA.body.userData.unhit(contact.fixtureB.body.userData)
            contact.fixtureB.body.userData.unhit(contact.fixtureA.body.userData)

    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass


class PlaceOccupied(b2QueryCallback):
    def __init__(self, ignore=[], ignore_type=[], dont_ignore=[]):
        b2QueryCallback.__init__(self)
        self.fixture = None
        self.ignore = ignore
        self.ignore_type = ignore_type
        self.dont_ignore = dont_ignore

    def ReportFixture(self, fixture):
        # Continue the query by returning True
        data = fixture.body.userData
        if not data in self.dont_ignore and (data in self.ignore or data.type in self.ignore_type):
            return True
        self.fixture = fixture
        return False # Stop query

class LidarCallback(b2RayCastCallback):
    """
    This class captures the closest hit shape.
    """
    def __init__(self, dont_report_type, dont_report_object, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.fraction = 1
        self.dont_report_type = dont_report_type
        self.dont_report_object = dont_report_object

    # Called for each fixture found in the query. You control how the ray proceeds
    # by returning a float that indicates the fractional length of the ray. By returning
    # 0, you set the ray length to zero. By returning the current fraction, you proceed
    # to find the closest point. By returning 1, you continue with the original ray
    # clipping.
    def ReportFixture(self, fixture, point, normal, fraction):
        if fixture.body.userData.type in self.dont_report_type or fixture.body.userData in self.dont_report_object or fixture.sensor:
            return -1
        self.fixture = fixture
        self.p2  = b2Vec2(point)
        self.normal = b2Vec2(normal)
        self.fraction = fraction
        # You will get this error: "TypeError: Swig director type mismatch in output value of type 'float32'"
        # without returning a value
        return fraction

class CheckObstacleRayCallback(LidarCallback):
    def __init__(self, dont_report, dont_report_object=list(), **kwargs):
        super().__init__(dont_report, dont_report_object, **kwargs)
        self.hit_obstacle = False

    # Called for each fixture found in the query. You control how the ray proceeds
    # by returning a float that indicates the fractional length of the ray. By returning
    # 0, you set the ray length to zero. By returning the current fraction, you proceed
    # to find the closest point. By returning 1, you continue with the original ray
    # clipping.
    def ReportFixture(self, fixture, point, normal, fraction):
        res = super().ReportFixture(fixture, point, normal, fraction)
        self.hit_obstacle = not (res == -1 or res == 1)
        if res == -1:
            return -1
        return res == 1
