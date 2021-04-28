from Box2D import b2QueryCallback, b2ContactListener


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

