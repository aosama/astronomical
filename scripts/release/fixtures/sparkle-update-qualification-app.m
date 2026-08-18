// Minimal AppKit host used to prove that Sparkle replaces and relaunches a real application bundle.

#import <AppKit/AppKit.h>

@interface AstronomicalUpdateQualificationDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AstronomicalUpdateQualificationDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification
{
    (void)notification;
    NSBundle *mainBundle = NSBundle.mainBundle;
    NSString *launchLogPath = [mainBundle objectForInfoDictionaryKey:@"AstronomicalQualificationLaunchLog"];
    NSString *processIdentifierPath = [mainBundle objectForInfoDictionaryKey:@"AstronomicalQualificationPIDFile"];
    NSString *bundleVersion = [mainBundle objectForInfoDictionaryKey:@"CFBundleVersion"];
    NSString *launchRecord = [NSString stringWithFormat:@"%@\n", bundleVersion];
    NSData *launchRecordData = [launchRecord dataUsingEncoding:NSUTF8StringEncoding];

    if (![[NSFileManager defaultManager] fileExistsAtPath:launchLogPath]) {
        [[NSData data] writeToFile:launchLogPath atomically:YES];
    }
    NSFileHandle *launchLog = [NSFileHandle fileHandleForWritingAtPath:launchLogPath];
    [launchLog seekToEndOfFile];
    [launchLog writeData:launchRecordData];
    [launchLog closeFile];

    NSString *processIdentifier = [NSString stringWithFormat:@"%d\n", NSProcessInfo.processInfo.processIdentifier];
    [processIdentifier writeToFile:processIdentifierPath atomically:YES encoding:NSUTF8StringEncoding error:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
}

@end

int main(void)
{
    @autoreleasepool {
        NSApplication *application = NSApplication.sharedApplication;
        AstronomicalUpdateQualificationDelegate *applicationDelegate =
            [[AstronomicalUpdateQualificationDelegate alloc] init];
        application.delegate = applicationDelegate;
        [application run];
    }
    return 0;
}
