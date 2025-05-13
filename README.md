git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
1 mvn clean package # Clean & build the project
2 mvn clean install # Clean, build & install in local repo
3 mvn deploy # Deploy to a remote repository

mvn clean install

https://rayisland.github.io/xyz/


selenium
 WebPageTest.java in the src/test/java directory.
 package org.test;
 import org.openqa.selenium.WebDriver;
 import org.openqa.selenium.chrome.ChromeDriver;
 import org.testng.Assert;
 import org.testng.annotations.AfterTest;
 import org.testng.annotations.BeforeTest;
 import org.testng.annotations.Test;

 import static org.testng.Assert.assertTrue;

 public class WebpageTest {
 private static WebDriver driver;
 @BeforeTest
 public void openBrowser() throws InterruptedException {
 driver = new ChromeDriver();
 driver.manage().window().maximize();
 Thread.sleep(2000);
 driver.get("https://rayisland.github.io/xyz/"); // 
 }

 @Test
 public void titleValidationTest(){
 String actualTitle = driver.getTitle();
 String expectedTitle = "Tripillar Solutions";
 Assert.assertEquals(actualTitle, expectedTitle);
 assertTrue(true, "Title should contain 'Tripillar'");
 }

 @AfterTest
 public void closeBrowser() throws InterruptedException {
 Thread.sleep(1000);
 driver.quit();
 }
