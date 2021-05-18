from Box2D import b2QueryCallback, b2ContactListener, b2RayCastCallback, b2Vec2


# collision handler
class ContactDetector(b2ContactListener):
    def __init__(self):
        super().__init__()

    def BeginContact(self, contact):
        contact.fixtureA.body.userData.hit_with.append(contact.fixtureB.body.userData)
        contact.fixtureB.body.userData.hit_with.append(contact.fixtureA.body.userData)
        #print('There was a contact!')

    def EndContact(self, contact):
        contact.fixtureA.body.userData.unhit(contact.fixtureB.body.userData)
        contact.fixtureB.body.userData.unhit(contact.fixtureA.body.userData)

    def PreSolve(self, contact, oldManifold):
        pass
    def PostSolve(self, contact, impulse):
        pass


class PlaceOccupied(b2QueryCallback):
    def __init__(self):
        b2QueryCallback.__init__(self)
        self.fixture = None

    def ReportFixture(self, fixture):
        # Continue the query by returning True
        self.fixture = fixture
        return False # Stop query

class LidarCallback(b2RayCastCallback):
    """
    This class captures the closest hit shape.
    """
    def __init__(self, **kwargs):
        b2RayCastCallback.__init__(self)
        self.fixture = None
        self.fraction = 1

    # Called for each fixture found in the query. You control how the ray proceeds
    # by returning a float that indicates the fractional length of the ray. By returning
    # 0, you set the ray length to zero. By returning the current fraction, you proceed
    # to find the closest point. By returning 1, you continue with the original ray
    # clipping.
    def ReportFixture(self, fixture, point, normal, fraction):
        self.fixture = fixture
        self.p2  = b2Vec2(point)
        self.normal = b2Vec2(normal)
        self.fraction = fraction
        # You will get this error: "TypeError: Swig director type mismatch in output value of type 'float32'"
        # without returning a value
        return fraction
