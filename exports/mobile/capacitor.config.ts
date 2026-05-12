import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.rfconnector.ai',
  appName: 'RF Connector AI',
  webDir: 'www',
  server: {
    androidScheme: 'https',
  },
  android: {
    webContentsDebuggingEnabled: true,
  },
  plugins: {
    Camera: {
      presentationStyle: 'fullScreen',
    },
  },
};

export default config;
