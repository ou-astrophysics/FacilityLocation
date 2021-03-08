import numpy as np
import numba
import logging
from numba.experimental import jitclass
import heapq
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

ConnectionMaskSpec = [
    ("inputMask", numba.int8[:, :]),
    ("mask", numba.int8[:, :]),
    ("dummyOffset", numba.int64),
]


@jitclass(ConnectionMaskSpec)
class ConnectionMask:
    def __init__(self, inputMask, trueIsAllowed=True, dummyOffset=1):
        # Mask.shape = (nC, nF)
        self.inputMask = ~inputMask if trueIsAllowed else inputMask
        self.mask = self.inputMask.copy().astype(np.int8)
        self.dummyOffset = dummyOffset

    def isConnectionAllowed(self, city, facility):
        return self.mask[city, facility] == 0

    def processConnect(self, city, facility):
        # If there are rows and columns to be treated as dummies,
        # ignore them.
        if facility < self.dummyOffset:
            return
        # Disallow connections between `facility` and any cities that
        # cannot connect to `city`.
        # Called when a city connects to a facility.
        self.mask[self.dummyOffset :, facility] += self.inputMask[
            city, self.dummyOffset :
        ]
        self.mask[facility, self.dummyOffset :] = self.mask[
            self.dummyOffset :, facility
        ]
        # Cannot override the input mask.
        self.mask |= self.inputMask

    def processDisconnect(self, city, facility):
        # If there are rows and columns to be treated as dummies,
        # ignore them.
        if facility < self.dummyOffset:
            return np.empty(shape=(0,), dtype=np.int64)
        # Re-allow connections between `facility` and any cities that
        # cannot connect to `city`.
        # Called when a city switches to another facility.
        oldMask = self.mask[self.dummyOffset :, facility].copy()

        self.mask[self.dummyOffset :, facility] = np.maximum(
            self.mask[self.dummyOffset :, facility]
            - self.inputMask[city, self.dummyOffset :],
            self.inputMask[city, self.dummyOffset :],
        )

        # Cannot override the input mask.
        self.mask |= self.inputMask

        # The mask should remain symmetric.
        self.mask[facility, self.dummyOffset :] = self.mask[
            self.dummyOffset :, facility
        ]
        # Return the list of cities that can now connect to this facility
        return np.flatnonzero(
            (self.mask[self.dummyOffset :, facility] <= 0) & (oldMask > 0)
        )


class Data:
    def __init__(self, dataShape, offersShape, logger):
        self.data = np.zeros(dataShape, dtype=np.float32)
        self.offers = np.empty(offersShape, dtype=np.float32)
        self.offers[:, :] = np.nan
        self.switchOffers = np.empty(offersShape, dtype=np.float32)
        self.switchOffers[:, :] = np.nan
        self.logger = logger

    def any(self, index):
        return np.any(self.data[:, index])

    def min(self, index):
        return np.nanmin(self.data[:, index])


class FData(Data):

    isOpen = 0
    isCandidate = 1
    inputs = 2
    numInputs = 3
    sumStandardInputs = 4
    sumSwitchInputs = 5
    isConnected = 6
    isCleanup = 7

    nameIndexKey = dict(
        isOpen=0,
        isCandidate=1,
        inputs=2,
        numInputs=3,
        sumStandardInputs=4,
        sumSwitchInputs=5,
        isConnected=6,
        isCleanup=7,
    )

    def __init__(self, nF, nC, logger=None):
        # Note: Change the 5 value if required
        super().__init__((nF, 8), (nF, nC), logger)
        self.alphas = []

    def addCandidate(self, alphaTuple):
        heapq.heappush(self.alphas, alphaTuple)

    def haveCandidates(self):
        return len(self.alphas) > 0

    def getMinAlphaCandidateInfo(self):
        return self.alphas[0]

    def getMinAlphaCandidate(self):
        # Return the *unopened* facility with the lowest
        # alpha. Discards any entries for open facilities
        # or connected cities.
        while self.haveCandidates():
            # Facility index is alphaTuple[1]
            alphaTuple = heapq.heappop(self.alphas)

            # Discard any facilites that are no longer candidates, have connected as cities to other facilities
            # or no longer have any valid offers.
            if (
                self.data[alphaTuple[1], FData.isCandidate]
                and not self.data[alphaTuple[1], FData.isConnected]
                and self.data[alphaTuple[1], FData.numInputs] > 0
            ):
                return alphaTuple

    def registerCandidate(self, facility, fCosts):
        # Note that the denominator corresponds to the number
        # of cities that will be *removed* from the *unconnected*
        # pool if the facility is opened. Therefore numInputs only
        # includes standard offers and not switch offers.
        self.data[facility, FData.sumStandardInputs] = np.nansum(
            self.offers[facility, :]
        )
        self.data[facility, FData.sumSwitchInputs] = np.nansum(
            self.switchOffers[facility, :]
        )

        alphaTuple = (
            (
                fCosts[facility]
                + self.data[facility, FData.sumStandardInputs]
                + self.data[facility, FData.sumSwitchInputs]
            )
            / self.data[facility, FData.numInputs],
            facility,
            self.data[facility, FData.numInputs],
        )

        self.data[facility, FData.isCandidate] = 1
        self.addCandidate(alphaTuple)

    def registerConnections(self, facility, cities):
        # Cities that have connected to a facility can never
        # become facilities themselves.
        self.data[cities, FData.isCandidate] = 0
        self.data[cities, FData.isConnected] = 1

        # Cities that connect to a facility should withdraw their
        # offers from other facilities, as should the facility being opened.
        citiesAndFacility = np.append(cities, facility)
        facsWithOffers = ~np.all(
            np.isnan(self.offers[:, citiesAndFacility])
            & np.isnan(self.switchOffers[:, citiesAndFacility]),
            axis=1,
        )
        offerIndexer = np.ix_(facsWithOffers, citiesAndFacility)
        self.offers[offerIndexer] = np.nan
        self.switchOffers[offerIndexer] = np.nan

        # Recompute input parameters for other cities
        self.data[facsWithOffers, FData.sumStandardInputs] = np.nansum(
            self.offers[facsWithOffers, :]
        )
        self.data[facsWithOffers, FData.sumSwitchInputs] = np.nansum(
            self.switchOffers[facsWithOffers, :]
        )
        self.data[facsWithOffers, FData.numInputs] -= 1


class CData(Data):

    isConnected = 0
    facility = 1
    didSwitch = 2
    output = 3

    notConnected = -2

    def __init__(self, nC, nF, logger=None):
        super().__init__((nC, 4), (nC, nF), logger)
        self.data[:, CData.facility] = CData.notConnected

    def registerConnections(self, cities, switchingCities, facility):
        allCities = np.append(cities, switchingCities).astype(int)
        self.data[allCities, CData.isConnected] = 1
        self.data[allCities, CData.facility] = facility
        self.data[switchingCities, CData.didSwitch] = 1
        return allCities

    def haveUnconnectedCities(self, haveDummy=1):
        return np.any(self.data[haveDummy:, CData.facility] == CData.notConnected)


class FacilityLocation:
    def __init__(self, fCosts, fcCosts, cfConnectionMask, logLevel=logging.DEBUG):
        self.fCosts = fCosts
        self.fcCosts = fcCosts
        self.cfConnectionMask = cfConnectionMask

        self.logger = logging.getLogger("FacilityLocation.debug")
        self.logger.setLevel(logLevel)

        self.fData = FData(*fcCosts.shape, self.logger)
        self.cData = CData(*fcCosts.T.shape, self.logger)
        self.connectionMask = ConnectionMask(cfConnectionMask)

    def connect(
        self,
        standardCities,
        facility,
        switchingCities=np.empty((0,), dtype=int),
        bypassSwitch=False,
    ):

        self.logger.info(
            f"Connecting cities {standardCities}...\nSwitching {switchingCities}..."
        )
        # Set city connection statuses.
        cities = self.cData.registerConnections(
            standardCities, switchingCities, facility
        )
        # Set facility connection statuses. Note: May not be required.
        self.logger.debug(
            f"Initial facility candidates: {self.fData.data[:, FData.isCandidate]}"
        )
        self.logger.debug(f"Connected cities: {cities}")
        self.fData.registerConnections(facility, cities)
        self.logger.debug(
            f"Updated facility candidates: {self.fData.data[:, FData.isCandidate]}"
        )

        if bypassSwitch:
            return

        # For any unopened facility that any of the connected cities
        # had made *standard* offers to, decrement the number of inputs.
        staleOfferFacilities, staleOfferFacilityCounts = np.unique(
            self.cData.data[cities, CData.facility][
                ~self.cData.data[cities, CData.didSwitch].astype(bool)
            ],
            return_counts=True,
        )

        self.fData.data[
            staleOfferFacilities.astype(int), FData.numInputs
        ] -= staleOfferFacilityCounts

        self.logger.info(f"Processing switches {cities}...")
        # For every city we just connected, process any offers it made
        # to *other* facilities and convert them to switch offers
        # *if possible*.
        currentOffers = self.cData.offers[cities, facility]
        # Compute the switch offer for all cities that just connected
        self.logger.info("Computing switch offers...")
        # Find cities that made a cheaper offer to another facility.
        # Note that valid offers can be zero, but a switch will only occur
        # if this is *strictly* cheaper than the current connection.
        numericOffers = np.nan_to_num(self.cData.offers[cities, :], nan=-1)
        switchOffers = np.where(
            numericOffers >= 0, (currentOffers - self.cData.offers[cities, :].T).T, 0
        )
        # Construct a mask exposing the possible switches for the cities
        # that just connected. These are facilities to which the
        # city has made an offer, excluding any facilities that are
        # already open.
        # Exclude the potential switches that are not *strictly* cheaper.
        # Note: If only one city is being connected, switchOffers will be 1D
        # but a 2D array is required.
        switchMask = np.atleast_2d((switchOffers > 0))

        # Obscure any facilities that are open.
        switchMask[:, self.fData.data[:, FData.isOpen].astype(bool)] = False

        self.logger.debug(
            f"cData.offers[cities, :].T =>\n {self.cData.offers[cities, :].T}\ncurrentOffers =>\n {currentOffers}\nswitchOffers =>\n {switchOffers}\nswitchMask => {switchMask}"
        )

        # Now compute a mask of facilities that each city can never connect to.
        # These are facilities that have already opened since the initial offer
        # was made, or facilities with a higher connection cost than the current
        # candidate.
        # Criteria are that an offer was made, but it was  more expensive than
        # the current one or that the switch candidate facility is now open.
        self.logger.info("Deriving impossible switches...")
        noSwitchMask = (numericOffers >= 0) & (
            switchOffers <= 0 | self.fData.data[:, FData.isOpen].astype(bool)
        )
        # Remove the current facility from the impossible switch mask because the
        # conection mask should not be released for the connection that was just
        # established.
        noSwitchMask[:, facility] = False
        self.logger.debug(
            f"np.atleast_2d(cData.offers[cities, :] > 0) =>\n{np.atleast_2d(numericOffers >= 0)}\nswitchOffers <= 0 =>\n{switchOffers <= 0}\nfData.data[:, FData.isOpen].astype(bool) =>\n {self.fData.data[:, FData.isOpen].astype(bool)}\nnoSwitchMask =>\n {noSwitchMask}"
        )

        # Update the switch offer records for the potentially switching cities
        self.cData.switchOffers[cities, facility] = np.nan
        self.cData.switchOffers[cities, :] = np.where(switchMask, switchOffers, np.nan)
        # Update the switch offer records for the switch candidate facilities
        self.fData.switchOffers[facility, cities] = np.nan
        self.fData.switchOffers[:, cities] = self.cData.switchOffers[cities, :].T

        # Now clear any standard offers that led to this facility opening.
        # Note that this does not clear *all* standard offers. Recorded
        # offers to other unopened facilities are still required.
        self.cData.offers[cities, facility] = np.nan
        self.fData.offers[facility, cities] = np.nan

        # Process the set of facilities that can never be switch candidates
        # for the connecting cities. If the impossible offer resulted in any connections
        # being masked, it is now safe to unmask them.
        # Note: Should this be done before or after making switch offers? I think before.
        # Also make an explicit offer for this connection.
        for cityFacilityPair in np.argwhere(noSwitchMask):
            self.logger.info(
                f"Forbidding the switch between city {cityFacilityPair[0]} and facility {cityFacilityPair[1]}"
            )
            enabledConnections = self.connectionMask.processDisconnect(
                *cityFacilityPair
            )
            for enabledCity in enabledConnections:
                # Note that the disallowed connection mask will be updated in offer()
                self.logger.debug(
                    f"Making offer from {enabledCity} to {cityFacilityPair[1]} after unmasking."
                )
                self.offer(
                    enabledCity,
                    cityFacilityPair[1],
                    self.fcCosts[cityFacilityPair[1], enabledCity],
                )

        # Process the set of facilities that the connecting cities have made switch
        # offers to.
        for facility in np.nonzero(switchMask & self.connectionMask.mask[cities, :])[1]:
            self.logger.debug(f"Registering {facility} as a switch candidate.")
            self.fData.data[facility, FData.numInputs] += 1
            self.fData.registerCandidate(facility, self.fCosts)
            for city in np.nonzero(
                switchMask[:, facility] & self.connectionMask.mask[cities, facility]
            )[0]:
                self.connectionMask.processConnect(city, facility)

    def offer(self, city, facility, cost):
        self.logger.info(f"Making offer from {city} to {facility}")

        # Increment the count of connections to facility
        # this number can be decremented later if
        # e.g. switches happen, but would be incremented on
        # either branch.
        self.fData.data[facility, FData.numInputs] += 1
        if self.fData.data[facility, FData.isOpen]:
            self.logger.info(
                f"Facility {facility} is already open. Connecting city {city}..."
            )
            # If the facility is already open then immediately connect.
            self.connect(city, facility, bypassSwitch=True)
            self.connectionMask.processConnect(city, facility)
        else:
            self.fData.data[facility, FData.isCandidate] = 1

            self.cData.offers[city, facility] = cost
            # Note: Do the fData and cData data stores need to be independent?
            self.fData.offers[facility, city] = cost

            self.fData.registerCandidate(facility, self.fCosts)
            # Update the disallowed connection mask at the point the offer
            # is made. This prevents inconsistent offers being registered.
            # If it is later discovered that this offered connection is
            # impossible, it will be possible to unmask any connections that
            # are disallowed at that stage.
            self.connectionMask.processConnect(city, facility)

    def openFacility(self, alpha, facility, cleanup=0):
        # Get the cities that have offers to facility
        offeringCities = np.flatnonzero(~np.isnan(self.fData.offers[facility]))
        switchingCities = np.flatnonzero(~np.isnan(self.fData.switchOffers[facility]))
        # Note: is this computation required? Only for printout.
        offeringCityCosts = [
            self.fcCosts[facility, offeringCity] for offeringCity in offeringCities
        ]
        switchingCityCosts = [
            self.fcCosts[facility, switchingCity] for switchingCity in switchingCities
        ]
        self.logger.info(f"Opening {facility}...\n{self.fData.offers}")
        self.logger.debug(
            f"({self.fCosts[facility]})\nStandard connections: {list(zip(offeringCities, offeringCityCosts))}\nSwitch connections: {list(zip(switchingCities, switchingCityCosts))}\nTotal cost = {self.fCosts[facility]+np.sum(offeringCityCosts) + np.sum(switchingCityCosts)}"
        )
        # Register that the facility is now open.
        self.fData.data[facility, FData.isCandidate] = 0
        self.fData.data[facility, FData.isOpen] = 1
        self.fData.data[facility, FData.isCleanup] = cleanup

        # Connect all offering cities to the facility.
        # At this point there should be no offers registered from
        # *connected* cities or between unconnected cities and
        # *open* facilities
        self.connect(offeringCities, facility, switchingCities)
        # Clear offers to facility
        self.fData.offers[facility] = np.nan
        self.cData.offers[:, facility] = np.nan

    def solve(self, fcPairs=None, haveDummy=False, allCitiesMustConnect=True):
        # If an explicit subset of pairs to consider is not passed then compute
        # a default set of all available pairs.
        # Only off-diagonal elements in the costs array need to be considered
        # cities cannot connect to themselves!
        if fcPairs is None:
            fcCostReducedLIndices = np.tril_indices_from(self.fcCosts, k=-1)
            fcCostReducedUIndices = np.triu_indices_from(self.fcCosts, k=1)
            fcPairs = np.array(
                [
                    np.append(fcCostReducedLIndices[0], fcCostReducedUIndices[0]),
                    np.append(fcCostReducedLIndices[1], fcCostReducedUIndices[1]),
                ]
            )  # 2xN

        # Explicitly filters any pairs with NaN connection costs.
        validFcPairs = fcPairs[:, ~np.isnan(self.fcCosts[tuple(fcPairs)])]  # 2xN_valid

        # sort the relevant costs into ascending order
        fcSortOrder = np.argsort(self.fcCosts[tuple(validFcPairs)])
        sortedCosts = self.fcCosts[tuple(validFcPairs)][fcSortOrder]
        # Compute the facility-city pair indices for the cost sorting order.
        sortedFcPairs = validFcPairs[:, fcSortOrder].T  # N_validx2 (sorted by cost)

        self.logger.debug(f"Pair costs:\n{list(zip(sortedFcPairs, sortedCosts))}\n")

        for cost, (facility, city) in zip(sortedCosts, sortedFcPairs):
            if city == 0 and haveDummy:
                continue
            self.logger.info(f"Processing: C -> {city}, F -> {facility}...")
            self.logger.debug(f"CCost -> {cost}, FCost -> {self.fCosts[facility]}")

            if self.cData.data[city, CData.isConnected]:
                # Cities that are already connected cannot make standard offers
                # to any other facility. They can make switch offers, but that
                # option is processed in offerSwitches().
                # Skip to next potential connection.
                self.logger.info(
                    f"City {city} is already connected to facility {self.cData.data[city, CData.facility]}"
                )
            elif self.connectionMask.isConnectionAllowed(city, facility):
                # Check whether any facility candidates exist.
                while self.fData.haveCandidates() and np.any(
                    self.fData.data[:, FData.isCandidate].astype(bool)
                    & ~self.fData.data[:, FData.isOpen].astype(bool)
                ):
                    self.logger.debug(
                        f"Have candidates...\n{np.flatnonzero(np.any(self.fData.data, axis=1))}\n"
                    )
                    # Extract current minimum alpha and corresponding facility
                    alpha, alphaFac, _ = self.fData.getMinAlphaCandidateInfo()
                    self.logger.debug(f"Alpha = {alpha}, Alpha Facility {alphaFac}")
                    self.logger.debug(
                        f"Candidates: {self.fData.data[:, FData.isCandidate]}"
                    )
                    if alpha > cost:
                        # If current minimum alpha exceeds the cost of this connection
                        # then continue accumulating offers and candidates...
                        self.logger.info(
                            f"Alpha ({alpha}) exceeds cost ({cost}). Continuing to accumulate candidates."
                        )
                        break

                    self.logger.info(
                        f"Alpha ({alpha}) is less than cost ({cost}). Opening next valid facility candidate..."
                    )
                    # convert a facility candidate into a facility
                    try:
                        self.openFacility(*self.fData.getMinAlphaCandidate()[:-1])
                    except TypeError as e:
                        if self.fData.haveCandidates():
                            self.logger.warning("Unexpected error: ", e)
                        else:
                            self.logger.info("No more candidates.")
                            break
                # Register an offer from a city to a facility
                self.offer(city, facility, cost)
            else:
                self.logger.info(
                    f"Connection between city {city} and facility {facility} is disallowed. No offer made."
                )

        self.logger.info("Finished processing city -> facility pairs.")

        # Continue opening facilities until all cities are connected to something
        # or the candidate list is exhausted.
        if allCitiesMustConnect:
            while (
                self.cData.haveUnconnectedCities(haveDummy=np.int64(haveDummy))
                and self.fData.haveCandidates()
            ):
                self.logger.info("Trying to connect left over cities...")
                try:
                    self.openFacility(*self.fData.getMinAlphaCandidate()[:-1], cleanup=1)
                except TypeError as e:
                    if self.fData.haveCandidates():
                        self.logger.warning("Unexpected error: ", e)
                    else:
                        self.logger.info("No more candidates.")
                        break
