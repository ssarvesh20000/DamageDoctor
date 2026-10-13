import XCTest

/// Smoke test: app launches and home screen is reachable.
final class DamageDoctorUITestsLaunchTests: XCTestCase {

    override class var runsForEachTargetApplicationUIConfiguration: Bool {
        true
    }

    func testLaunchShowsHomeScreen() throws {
        let app = XCUIApplication()
        app.launchArguments += ["-UITesting"]
        app.launch()

        XCTAssertTrue(app.staticTexts["DamageDoctor"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["choosePhotoButton"].exists)
        XCTAssertTrue(app.buttons["takePhotoButton"].exists)

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Launch Screen"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
